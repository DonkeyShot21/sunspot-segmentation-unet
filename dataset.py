from __future__ import print_function, division
from datetime import datetime
from datetime import timedelta
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.time import parse_time
from sunpy.coordinates import frames
from sunpy.physics.differential_rotation import solar_rotate_coordinate

from astropy.units import Quantity
from sunpy.map import Map

from sunpy.net.vso import VSOClient
from sunpy.net.hek2vso import hek2vso, H2VClient

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from torchvision.transforms import Compose
import math

import warnings
warnings.filterwarnings(action='once')





def search_VSO(start_time, end_time):
    client = VSOClient()
    query_response = client.query_legacy(tstart=start_time,
                                         tend=end_time,
                                         instrument='HMI',
                                         physobs='intensity',
                                         sample=3600)
    results = client.fetch(query_response[:1],
                           path='./tmp/{file}',
                           site='rob')
    continuum_file = results.wait()

    query_response = client.query_legacy(tstart=start_time,
                                         tend=end_time,
                                         instrument='HMI',
                                         physobs='los_magnetic_field',
                                         sample=3600)
    results = client.fetch(query_response[:1],
                           path='./tmp/{file}',
                           site='rob')
    magnetic_file = results.wait()
    return continuum_file[0], magnetic_file[0]

def normalize_map(map):
    img = map.data
    img[np.isnan(img)] = 0
    img_min = np.amin(img)
    img_max = np.amax(img)
    return (img - img_min) / (img_max - img_min)

def rotate_coord(map, coord, date):
    coord_sc = SkyCoord(
        [(float(v[1]),float(v[0])) * u.deg for v in np.array(coord)],
        obstime=date,
        frame=frames.HeliographicCarrington)
    coord_sc = coord_sc.transform_to(frames.Helioprojective)
    rotated_coord_sc = solar_rotate_coordinate(coord_sc, map.date)

    px = map.world_to_pixel(rotated_coord_sc)
    return [(int(px.x[i].value),int(px.y[i].value)) for i in range(len(px.x))]

def remove_if_exists(file):
    if os.path.exists(file):
        os.remove(file)

def show_mask(img, mask):
    img = (255 * img).astype(np.uint8)
    mask = np.dstack((mask,mask,mask))
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    im2,contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        cv2.drawContours(img, contour, -1, (255,0,0), thickness = 1)
    img = np.dstack((img,img,img))
    b,g,r = cv2.split(img)
    r = cv2.add(b, 30, dst = b, mask = binary, dtype = cv2.CV_8U)
    cv2.merge((b,g,r), img)
    Image.fromarray(img).show()




class HelioDataset(Dataset):
    def __init__(self, SIDC_filename, fenyi_filename, n_samples):
        super(Dataset, self).__init__()
        self.n_samples = n_samples

        sidc_csv = pd.read_csv(SIDC_filename, sep=';', header=None)
        sidc_csv.drop(sidc_csv[[3,5,6,7]], axis=1, inplace=True)
        sidc_csv.astype(np.int32)
        self.sidc_csv = sidc_csv[sidc_csv[0] == 2014]

        self.fenyi_sunspot = pd.read_csv(fenyi_filename, sep=',')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # sampling with probability from SIDC
        print("Sampling from SIDC...")
        row = self.sidc_csv.sample(weights=self.sidc_csv[4])
        day = '/'.join(map(str, row.iloc[0][:-1]))
        date = datetime.strptime(day + ' 12:00:00', '%Y/%m/%d %H:%M:%S')

        # loading sunspot data from DPD
        print("Loading sunspot data...")
        dpd = self.fenyi_sunspot.query(("year == @date.year & "
                                        "month == @date.month & "
                                        "day == @date.day"))

        time = datetime.strptime('-'.join([str(i) for i in list(dpd.iloc[0])[1:7]]), '%Y-%m-%d-%H-%M-%S')
        start_time = (time - timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')
        end_time = (time + timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')

        try:
            print("Searching VSO...")
            continuum_file, magnetic_file = search_VSO(start_time, end_time)
            hmi_cont = Map(continuum_file)
            hmi_mag = Map(magnetic_file)
        except Exception as e:
            print(e)
            remove_if_exists(continuum_file)
            remove_if_exists(magnetic_file)
            return self.__getitem__(idx)

        # get the data from the maps
        img_cont = normalize_map(hmi_cont)
        img_mag = 2 * normalize_map(hmi_mag) - 1
        inputs = np.dstack((img_cont, img_mag))

        # get the coordinates and the date of the sunspots from DPD
        print("Creating mask...")
        ss_coord = dpd[['heliographic_latitude', 'heliographic_longitude']]
        ss_date = parse_time(time)
        sunspots = rotate_coord(hmi_cont, ss_coord, ss_date)

        # mask = (255 * img_cont).astype(np.uint8)
        mask = np.zeros(img_cont.shape, dtype=np.float32)

        ws = dpd[['projected_whole_spot',
                  'group_number',
                  'group_spot_number']]

        for index, row in ws.iterrows():
            wsa = row['projected_whole_spot']
            if wsa < 0:
                match = ws.query(("group_number == @row.group_number & "
                                  "group_spot_number == -@wsa"))
                area = match['projected_whole_spot'].iloc[0]
                ws.loc[row.name,'projected_whole_spot'] = area

        groups = list(ws['group_number'].unique())
        disk_mask = np.where(255*img_cont > 15)
        disk_mask = {(c[0],c[1]) for c in np.column_stack(disk_mask)}
        disk_mask_num_px = len(disk_mask)
        whole_spot_mask = set()

        for i in range(len(sunspots)):
            o = 4 # offset
            p = sunspots[i]
            # g_number = groups.index(ws.iloc[i]['group_number'])
            group = img_cont[int(p[1])-o:int(p[1])+o,int(p[0])-o:int(p[0])+o]
            low = np.where(group == np.amin(group))

            center = (img_cont.shape[0] / 2, img_cont.shape[1] / 2)
            distance = np.linalg.norm(tuple(j-k for j,k in zip(center,p)))
            cosine_amplifier = math.cos(math.radians(1) * distance / center[0])
            norm_num_px = cosine_amplifier * ws.iloc[i]['projected_whole_spot']
            ss_num_px = 8.7 * norm_num_px * disk_mask_num_px / 10e6

            print(center, distance, cosine_amplifier, norm_num_px, ss_num_px)

            new = set([(p[1] - o + low[1][0], p[0] - o + low[0][0])])
            whole_spot = set()
            candidates = dict()
            expansion_rate = 3
            while len(whole_spot) < ss_num_px:
                expand = {(n[0]+i,n[1]+j)
                          for i in [-1,0,1]
                          for j in [-1,0,1]
                          for n in new}
                for e in set(expand - whole_spot):
                    candidates[e] = img_cont[e]
                new = sorted(candidates, key=candidates.get)[:expansion_rate]
                for n in new:
                    candidates.pop(n, None)
                whole_spot.update(set(new))

            whole_spot_mask.update(whole_spot)

        print(whole_spot_mask - disk_mask)
        for c in set.intersection(whole_spot_mask, disk_mask):
            mask[c] = 1

        # show_mask(img_cont, mask)

        remove_if_exists(continuum_file)
        remove_if_exists(magnetic_file)

        data_pair = {'img': torch.from_numpy(inputs), 'mask': torch.from_numpy(mask)}
        return data_pair






if __name__ == '__main__':

    dataset = HelioDataset('./data/SIDC_dataset.csv',
                           'data/sDPD2014.txt',
                           10)

    data_loader = DataLoader(dataset)

    for idx, batch_data in enumerate(data_loader):
        print(idx)
        print(batch_data['img'].size())
        print(batch_data['mask'].size())







## QUERYING AND DRAWINF HEK


# fig = plt.figure()
# ax = plt.subplot(projection=hmi)
# hmi.plot(axes=ax)
#
# for ss in responses:
#     p = [v.split(" ") for v in ss["hpc_boundcc"][9:-2].split(',')]
#     ss_date = parse_time(ss['event_starttime'])
#
#     ss_boundary = SkyCoord(
#         [(float(v[0]), float(v[1])) * u.arcsec for v in p],
#         obstime=ss_date,
#         frame=frames.Helioprojective)
#     rotated_ss_boundary = solar_rotate_coordinate(ss_boundary, hmi.date)
#
#     px = hmi.world_to_pixel(rotated_ss_boundary)
#     points = [[(int(px.x[i].value),int(px.y[i].value)) for i in range(len(px.x))]]
#     cv2.fillPoly(img, np.array(points), 127)
#
#
#     # ax.fill(hmi.world_to_pixel(rotated_ss_boundary),'b')
#     ax.plot_coord(rotated_ss_boundary, color='c')
#
# ax.set_title('{:s}\n{:s}'.format(hmi.name, ss['frm_specificid']))
# plt.colorbar()
# plt.show()


# 0.2650327216254339 3.795

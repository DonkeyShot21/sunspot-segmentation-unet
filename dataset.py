from __future__ import print_function, division
import datetime as dt
from datetime import datetime
from datetime import timedelta
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.net import hek
from sunpy.time import parse_time
from sunpy.coordinates import frames
from sunpy.physics.differential_rotation import solar_rotate_coordinate

from sunpy.net.helioviewer import HelioviewerClient
from astropy.units import Quantity
from sunpy.map import Map

from sunpy.net.vso import VSOClient
from sunpy.net.hek2vso import hek2vso, H2VClient

import pickle as pkl
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from torchvision.transforms import Compose
import math

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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
        start_time = (time - dt.timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')
        end_time = (time + dt.timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%S')

        try:
            print("Searching VSO...")
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

            # load maps
            hmi_cont = Map(continuum_file[0])
            hmi_mag = Map(magnetic_file[0])

        except:
            if os.path.exists(continuum_file[0]):
                os.remove(continuum_file[0])
            if os.path.exists(magnetic_file[0]):
                os.remove(magnetic_file[0])
            return self.__getitem__(idx)

        # get the data from the maps
        img_cont = hmi_cont.data
        img_cont[np.isnan(img_cont)] = 0
        img_min = np.amin(img_cont)
        img_max = np.amax(img_cont)
        img_cont = (img_cont - img_min) / (img_max - img_min)

        img_mag = hmi_mag.data
        img_cont[np.isnan(img_mag)] = 0
        img_min = np.amin(img_mag)
        img_max = np.amax(img_mag)
        img_mag = 2 * (img_mag - img_min) / (img_max - img_min) - 1

        inputs = np.dstack((img_cont, img_mag))

        # get the coordinates and the date of the sunspots from DPD
        print("Creating mask...")
        ss_coord = dpd[['heliographic_latitude', 'heliographic_longitude']]

        date = dpd[['year','month','day','hour','minute','second']].iloc[0]
        date = datetime.strptime('-'.join([str(i) for i in list(date)]),
                                 '%Y-%m-%d-%H-%M-%S')
        date = parse_time(date)

        ss_boundary = SkyCoord(
            [(float(v[1]),float(v[0])) * u.deg for v in np.array(ss_coord)],
            obstime=date,
            frame=frames.HeliographicCarrington)
        ss_boundary = ss_boundary.transform_to(frames.Helioprojective)
        rotated_ss_boundary = solar_rotate_coordinate(ss_boundary, hmi_cont.date)

        px = hmi_cont.world_to_pixel(rotated_ss_boundary)
        points = [(int(px.x[i].value),int(px.y[i].value)) for i in range(len(px.x))]

        # mask = (255 * img_cont).astype(np.uint8)
        mask = np.zeros(img_cont.shape, dtype=np.uint8)
        img_cont = (255 * img_cont).astype(np.uint8)

        ws = dpd[['projected_whole_spot',
                  'corrected_projected_whole_spot',
                  'group_number',
                  'group_spot_number']]

        for index, row in ws.iterrows():
            for k in ['projected_whole_spot','corrected_projected_whole_spot']:
                wsa = row[k]
                if wsa < 0:
                    match = ws.query(("group_number == @row.group_number & "
                                      "group_spot_number == -@wsa"))
                    area = match[k].iloc[0]
                    ws.loc[row.name,k] = area

        groups = list(ws['group_number'].unique())
        disk_px = len(np.where(255*img_cont > 15)[0])
        whole_spot_mask = []

        for i in range(len(points)):
            o = 4
            # o = max(35, 2*int(math.sqrt(ws.iloc[i]['projected_whole_spot'])))
            p = points[i]
            g = groups.index(ws.iloc[i]['group_number'])
            # cv2.rectangle(mask, (int(p[0]) - o, int(p[1]) - o),
            #              (int(p[0]) + o, int(p[1]) + o), g+1, 1)

            group = img_cont[int(p[1]) - o : int(p[1]) + o,
                             int(p[0]) - o : int(p[0]) + o]

            low_group_coord = np.where(group == np.amin(group))
            low = (p[1] - o + low_group_coord[1][0],
                   p[0] - o + low_group_coord[0][0])

            whole_spot = set([low])
            candidates = dict()
            new = low

            half_loc = img_cont.shape[0] / 2
            distance = np.linalg.norm(tuple(i-j for i,j in zip((half_loc,half_loc),p))) / half_loc
            print("distance",distance)

            cos_amplifier = math.cos(math.radians(1) * distance)
            print("amplifier", cos_amplifier)

            max_num_px = cos_amplifier * 8 * ws.iloc[i]['projected_whole_spot'] * disk_px / 10e6
            print("max", max_num_px)

            while len(whole_spot) < max_num_px:
                expand = {(new[0]+i,new[1]+j)
                          for i in range(-1,2)
                            for j in range(-1,2)}
                for e in set(expand - whole_spot):
                    candidates[e] = img_cont[e]
                new = min(candidates, key=candidates.get)
                candidates.pop(new, None)
                whole_spot.add(new)

            whole_spot_mask += list(whole_spot)

        for c in whole_spot_mask:
            mask[c] = 254

        mask = np.dstack((mask,mask,mask))
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
        im2,contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            cv2.drawContours(img_cont, contour, -1, (255,0,0), thickness = 1)

        img_cont = np.dstack((img_cont,img_cont,img_cont))
        b,g,r = cv2.split(img_cont)

        # add a constant to R channel to highlight the selected area in reed
        r = cv2.add(b, 30, dst = b, mask = binary, dtype = cv2.CV_8U)

        # merge the channels back together
        cv2.merge((b,g,r), img_cont)

        # for g in range(len(groups)):
        #     coords = np.where(mask == g+1)
        #     minx, maxx = np.amin(coords[1]), np.amax(coords[1])
        #     miny, maxy = np.amin(coords[0]), np.amax(coords[0])
        #     cv2.rectangle(mask, (minx,miny), (maxx,maxy), 255, 2)
        #
        #     group = (255 * img_cont[miny:maxy,minx:maxx]).astype(np.uint8)
            # Image.fromarray(group).show()



        Image.fromarray(img_cont).show()

        data_pair = {'img': torch.from_numpy(inputs), 'mask': torch.from_numpy(mask)}

        if os.path.exists(continuum_file[0]):
            os.remove(continuum_file[0])
        if os.path.exists(magnetic_file[0]):
            os.remove(magnetic_file[0])
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

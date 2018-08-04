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


class HelioDataset(Dataset):
    def __init__(self, SIDC_filename, fenyi_filename):
        super(Dataset, self).__init__()

        sidc_csv = pd.read_csv(SIDC_filename, sep=';', header=None)
        sidc_csv.drop(sidc_csv[[3,5,6,7]], axis=1, inplace=True)
        sidc_csv.astype(np.int32)
        self.sidc_csv = sidc_csv[sidc_csv[0] == 2014]

        self.fenyi_sunspot = pd.read_csv(fenyi_filename, sep=',')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        row = self.sidc_csv.sample(weights=self.sidc_csv[4])
        day = '/'.join(map(str, row.iloc[0][:-1]))
        # start_time = day + ' 12:00:00'
        # end_time = day + ' 13:00:00'
        date = datetime.strptime(day + ' 12:00:00', '%Y/%m/%d %H:%M:%S')

        dpd = self.fenyi_sunspot.query(("year == @date.year & "
                                        "month == @date.month & "
                                        "day == @date.day"))
        date = datetime.strptime('-'.join([str(i) for i in list(dpd.iloc[0])[1:7]]), '%Y-%m-%d-%H-%M-%S')


        # Image path
        print("Searching HEK...")
        hek_client = hek.HEKClient()
        responses = hek_client.search(hek.attrs.Time(date, date + dt.timedelta(hours=12)),
                                      hek.attrs.SS)


        pkl.dump(responses, open('responses.pkl', 'wb'))
        # responses = pkl.load(open('responses.pkl', 'rb'))

        print("Searching VSO...")
        client = VSOClient()

        query_response = client.query_legacy(tstart=responses[0]['event_starttime'],
                                             tend=responses[0]['event_endtime'],
                                             instrument='HMI',
                                             physobs='los_magnetic_field',
                                             sample=3600)
        results = client.fetch(query_response[:1], path='./tmp/{file}', site='rob')
        files = results.wait()

        query_response = client.query_legacy(tstart=responses[0]['event_starttime'],
                                             tend=responses[0]['event_endtime'],
                                             instrument='HMI',
                                             physobs='intensity',
                                             sample=3600)
        results = client.fetch(query_response[:1], path='./tmp/{file}', site='rob')
        files.append(results.wait())

        print(files)

        for f in files:
            print(f)
            hmi = Map(f)
            fig = plt.figure()
            ax = plt.subplot(projection=hmi)
            hmi.plot(axes=ax)

            img = hmi.data
            img[np.isnan(img)] = 0

            img_min = np.amin(img)
            img_max = np.amax(img)

            img = (255 * (img - img_min) / (img_max - img_min)).astype(np.uint8)
            print(img.shape)


            for _,row in dpd.iterrows():
                r = [str(i) for i in list(row)[1:7]]
                ss_coord = list(row[-5:-3])
                lol = datetime.strptime('-'.join(r), '%Y-%m-%d-%H-%M-%S')
                ss_date = parse_time(lol)
                ss_boundary = SkyCoord(
                    [tuple([float(v) * u.deg for v in ss_coord[::-1]])],
                    obstime=hmi.date,
                    frame=frames.HeliographicCarrington)
                ss_boundary.transform_to(frames.Helioprojective)
                # rotated_ss_boundary = solar_rotate_coordinate(ss_boundary, hmi.date)

                px = hmi.world_to_pixel(ss_boundary)
                points = [(int(px.x[i].value),int(px.y[i].value)) for i in range(len(px.x))]
                print(points)
                cv2.rectangle(img, (int(px.x[0].value)-50,int(px.y[0].value)-50),
                              (int(px.x[0].value)+50,int(px.y[0].value)+50), 255)

            for ss in responses:
                p = [v.split(" ") for v in ss["hpc_boundcc"][9:-2].split(',')]
                ss_date = parse_time(ss['event_starttime'])

                ss_boundary = SkyCoord(
                    [(float(v[0]), float(v[1])) * u.arcsec for v in p],
                    obstime=ss_date,
                    frame=frames.Helioprojective)
                rotated_ss_boundary = solar_rotate_coordinate(ss_boundary, hmi.date)

                px = hmi.world_to_pixel(rotated_ss_boundary)
                points = [[(int(px.x[i].value),int(px.y[i].value)) for i in range(len(px.x))]]
                cv2.fillPoly(img, np.array(points), 127)


                # ax.fill(hmi.world_to_pixel(rotated_ss_boundary),'b')
                ax.plot_coord(rotated_ss_boundary, color='c')
            Image.fromarray(img).show()



        ax.set_title('{:s}\n{:s}'.format(hmi.name, ss['frm_specificid']))
        plt.colorbar()
        plt.show()

        data_pair = {'img': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
        return data_pair


if __name__ == '__main__':

    dataset = HelioDataset('./data/SIDC_dataset.csv', 'data/sDPD2014.txt')

    data_loader = DataLoader(dataset)

    for idx, batch_data in enumerate(data_loader):
        print(idx)
        print(batch_data['img'].size())
        print(batch_data['mask'].size())

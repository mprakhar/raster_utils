# -*- coding: utf-8 -*-
#!/usr/bin/env python
import infoFinder as info
import shp_rstr_stat as srs
import numpy as np
from osgeo import gdal, gdalnumeric, ogr, osr
import my_math as mth


# Raster > Raster_file > Image_Arr



class Raster(object):

    def _init_(self):
        self.path = None
        self.sat = None
        self.prod = None
        self.sample = None
        self.georef = None

    def mean_annual_img(self, input_zone_polygon, start, end):
        sum0 = np.empty(np.shape(srs.zone_mask(input_zone_polygon, self.filenamelist[0][0])))
        mask = srs.zone_mask(input_zone_polygon, self.filenamelist[0][1])
        for year in range(start, end):
            sum = sum0
            i = 0
            for input_value_raster in self.filenamelist():
                yearm = self.yearfinder(input_value_raster)
                print yearm
                if int(int(yearm) / 100) == year:
                    print 'true'
                    sum = sum + srs.zone_mask(input_zone_polygon, input_value_raster)[0]
                    i = i + 1
            avg = (sum / i) * mask  # to get rid of masked value we multiply by mask
            srs.arr_to_raster(avg, self.georef, self.sat + self.prod + str(year) + ".tif")

    def try_func(self):
        print 'under construction'

    def try_func2(self):
        self.try_func()

    def filenamelist(self, ext):
        return info.filenamefinder(self.path, self.sat, self.prod, ext)

    def f1(self):
        print 'f1'
    def f2(self):
        self.f1()




class Raster_file(Raster):
    def _init_(self):
        self.file = None

    def zone_mask(self, input_zone_polygon):
        imgarray, datamask = srs.zone_mask(input_zone_polygon, self.file)
        return imgarray, datamask

    def yearfinder(self, ):
        return info.yearfinder(self.file, self.sat)

    def yeardayfinder(self):
        return info.yeardayfinder(self.file, self.sat)

    def raster_as_array(self):
        # Open data
        driver = gdal.GetDriverByName('ENVI')  # envi
        driver.Register()
        raster = gdal.Open(self.file)
        banddataraster = raster.GetRasterBand(1)
        raster_array = banddataraster.ReadAsArray().astype(np.float)
        return raster_array

    def zonal_stat(self, input_zone_polygon, ID ):
        return srs.loop_zonal_stats(input_zone_polygon, self.file, ID)



class Image_arr(Raster_file):

    def __init__(self, array):
        #Raster._init_()
        self.img_array = array

    def RosinThreshold(self, MINVAL):
        return mth.RosinThreshold(self, MINVAL)

    def arr_to_raster(self, out_raster):
        srs.arr_to_raster(self.img_array, self.georef, out_raster)

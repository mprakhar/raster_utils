#__author__ = 'Prakhar MISRA'
# Created 2/29/2016
#Last edit 2/29/2016

# Purpose: To return info about various types as mentioned below
# Location of output: -

import os, sys, fnmatch
import glob
from glob import glob
from datetime import timedelta, date
from dateutil import rrule



#search for raster files in the desired folder
def findRasters (path, filter):
    for root, dirs, files in os.walk(path, filter):
        for file in fnmatch.filter(files, filter):
            yield os.path.join (root, file)

#search for the shape files in the source folder
def findVectors (path, filter):
    for root, dirs, files in os.walk(path, filter):
        for file in fnmatch.filter(files, filter):
            yield os.path.join (root, file)

# Find the filenames of the  sat and product
def filenamefinder(raster_path, sat, prod, ext):

    if sat in ['MODIS','OMI']:
            sufx = '' if sat == 'MODIS' else 'ave'
            filename = glob(os.path.join(raster_path, '*.'+prod+'*'+sufx+'.Global'))
            return filename
    # if sat in ['MODISb']: # b denotes beta; custom satellite filename
    #     filename = glob(os.path.join(raster_path, '*.'+prod+'*'+'.tif'))
    #     return filename
    # if sat in ['OMIb']: # b denotes beta; custom satellite filename
    #     filename = glob(os.path.join(raster_path, '*.'+prod+'*'+'.tif'))
    #     return filename
    if sat in['VIIRS']:
            filename = glob(os.path.join(raster_path, '*'+ext))
            return filename
    if sat in['DMSP']:
            filename = glob(os.path.join(raster_path, '*'+'stable'+'*'+ext))
            return filename

    if sat in ['LSWC']:
            filename = glob(os.path.join(raster_path, 'LSWC' + '????????'))
            return filename

    else:
        filename = glob(os.path.join(raster_path, '*'+ext))
        return filename


     # this filename is actually a list containin names of all relevant files

# find the year and month as YYYYMM for monthly data e.g. MODIS, OMI and YYYY for annual data e.g. OMI, DMSP on those filename
def yearfinder(raster_name, sat):
    f_name= os.path.split(raster_name)[1]
    if sat=='MODIS':
        year= f_name[-17:-11]
    # if sat=='MODISb':
    #     year= f_name[-17:-11]
    if sat=='VIIRS':
        year = f_name[10:16]
    # if sat=='VIIRSb': # refers to beta images of VIIRS like those published in 2013
    #     year = f_name[5:11]
    if sat=='OMI':
        year=f_name[-17:-11]
    # if sat=='OMIb':
    #     year=f_name[-17:-11]
    if sat=='DMSP':
        year=int(f_name[3:7])*100
    if sat == 'cust': # cust refers to custom file names
        year = int(f_name[-8:-4])
    if sat == 'LSWC':
        year = f_name[4:12]
    return year

# Find the year and the day of year from daily level images
def yeardayfinder(raster_name, sat):
    f_name= os.path.split(raster_name)[1]
    if sat=='MODIS':
        year= f_name[-18:-14]
        day = f_name[-14:-11]
    # if sat=='MODISb':
    #     year= f_name[-17:-11]
    if sat=='OMI':
        year=f_name[-15:-11]
        mon = f_name[-11:-9]
        day_of_mon = f_name[-9:-7]
        ddate = date(int(year), int(mon), int(day_of_mon)) # converting yyyymmdd format to yyyyddd format
        day = ddate.timetuple().tm_yday

    return [year, day]

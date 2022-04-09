# __author__ = 'Prakhar MISRA'
# Created 8/24/2016
# Last edit 8/28/2016

#Purpose: (1) to generate monthly mean images from daily data by removing 2%-98% outliers nad taking mean
#         (2) to generate annual images from monthly images
#

#Output expected:

import numpy as np
from numpy import *
import os
import os.path
import coord_translate as ct
import gdal
from datetime import timedelta, date
from dateutil import rrule
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from glob import glob

#Pvt imports
import shp_rstr_stat as srs
import my_math as mth
import infoFinder as info

# Input
gb_path = r'/home/prakhar/Research/AQM_research//'    # global path tp be appended to each path

# Output
img_save_path = gb_path + r'/Data/Data_process//'



# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task I  Clean mean monthly images     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

def mean_month(prodT, start_date, end_date, minmax=False, weight = False):
    # please note that prodT is an object of classRaster
    # the function deliberately has end_date so that not only for a single month but also seasonal means can be taken up
    a =0
    for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):

        print date_m
        mon1 = date_m       # start of month
        mon2 = date_m + relativedelta(months=1) - relativedelta(days=1)     # end of month

        ls = []

        for date_d in rrule.rrule(rrule.DAILY,dtstart=mon1, until=mon2 ):
            for prodT.file in glob(os.path.join(prodT.path, '*.'+prodT.prod+'*'+'.Global')) :

                # fnd integer list of  year and daycount
                date_y_d = map(int, prodT.yeardayfinder())
                # if (date_y_d == [date_d.year,date_d.timetuple().tm_yday ]) & os.path.isfile(prodT.file): # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue

                # find file with same date and convert to array and append
                if (date_y_d == [date_d.year, date_d.timetuple().tm_yday]) & os.path.isfile(prodT.file):  # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue
                    # print 'file found'
                    try:
                        prod_arr = prodT.raster_as_array()
                        # print date_d, prod_arr[pix[1], pix[0]]
                        ls.append(prod_arr)
                    except AttributeError:
                        print ' some error in ',  date_d

        # Calculate mean
        ls_arr = np.array(ls)

        # get rid of all zero values
        # ls_arr[ls_arr == 0] = np.nan

        # get rid of too high or too values
        # ls_98 = np.percentile(ls_arr, 98.0,0)       # kya haga tha pehle.. this hsould have been axis=0 all along
        # ls_02 = np.percentile(ls_arr, 02.0,0)
        # ls_arr = np.where(ls_arr>=ls_98, ls_98, ls_arr)       # <<<< This used to np.nan instrad of ls_98 earlier. is this a fuck up??
        # ls_arr = np.where(ls_arr <= ls_02, ls_02, ls_arr)
        ls_mean = np.nanmean(ls_arr, axis=0)

        if weight!=False:
            ls_mean = np.average(ls_arr, axis=0, weights = weight)

        if minmax:
            return ls_mean #, ls_02, ls_98
        else :
            return ls_mean


# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task II  Clean mean monthly images with sampling    *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

def sampledmean_month(prodT, start_date, end_date, minmax=False):
    # minmax option can give give the mean between 2% to 98%
    # the function deliberately has end_date so that not only for a single month but also seasonal means can be taken up
        a =0
        for date_m in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date):

            print date_m
            mon1 = date_m       # start of month
            mon2 = date_m + relativedelta(months=1) - relativedelta(days=1)     # end of month

            #minimum num obseservations neede each month. CInterval =25, CL = 95, populaiton = 30
            min_obs = 10

            ls = []

            for date_d in rrule.rrule(rrule.DAILY,dtstart=mon1, until=mon2 ):
                for prodT.file in glob(os.path.join(prodT.path, '*.'+prodT.prod+'*'+'.Global')) :

                    # fnd integer list of  year and daycount
                    date_y_d = map(int, prodT.yeardayfinder())
                    # if (date_y_d == [date_d.year,date_d.timetuple().tm_yday ]) & os.path.isfile(prodT.file): # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue

                    # find file with same date and convert to array and append
                    if (date_y_d == [date_d.year, date_d.timetuple().tm_yday]) & os.path.isfile(prodT.file):
                        # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue

                        # print 'file found'
                        try:
                            prod_arr = prodT.raster_as_array()
                            # print date_d, prod_arr[pix[1], pix[0]]
                            ls.append(prod_arr)
                        except AttributeError:
                            print ' some error in ',  date_d

            # Calculate mean
            ls_arr = np.array(ls)
            ls_arr[ls_arr == 0] = np.nan
            ls_mean = np.nanmean(ls_arr, axis=0)

            # sampling step
            # cleaning those monthly means where num of observations <= min_obs
            obs_arr = np.array(ls)
            ls_count = mth.count_nonzero(obs_arr, axis=0)
            ls_count[ls_count < min_obs] = 0
            ls_count[ls_count >= min_obs] = 1

            return ls_mean*ls_count



# * * * *  * * # * * * *  * * # * * * *  * *# # * *      Task III  Clean mean annual image from monthly image     *  * * # * * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

def mean_year(prodT, start_date, end_date, minmax=False):

# the function deliberately has end_date so that not only for a single month but also seasonal means can be taken up
        a =0
        for date_m in rrule.rrule(rrule.YEARLY, dtstart=start_date, until=start_date):

            print date_m
            mon1 = date_m       # start of year
            mon2 = date_m + relativedelta(years=1) - relativedelta(days=1)     # end of year

            ls = []

            for date_d in rrule.rrule(rrule.MONTHLY,dtstart=mon1, until=mon2 ):
                for prodT.file in glob(os.path.join(prodT.path, '*.'+prodT.prod+'*'+'.Global')) :

                    # fnd integer list of  year and month
                    date_y_d = prodT.yearfinder()
                    # if (date_y_d == [date_d.year,date_d.timetuple().tm_yday ]) & os.path.isfile(prodT.file): # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue

                    # find file with same date and convert to array and append
                    if (date_y_d == (str(date_d.year) + str('%02d' % date_d.month))) & os.path.isfile(prodT.file):  # [year,day_count] comparison; 2006228-2006301 data missing for OMI; 20131005OMINO2-20131231, 20150803 some issue
                        # print 'file found'
                        try:
                            prod_arr = prodT.raster_as_array()
                            # print date_d, prod_arr[pix[1], pix[0]]
                            ls.append(prod_arr)
                        except AttributeError:
                            print ' some error in ',  date_d
                    # else:
                    #     print date_d, ' file not found'

            # Calculate mean
            ls_arr = np.array(ls)
            ls_mean = np.nanmean(ls_arr, axis=0)
            if minmax:
                return ls_mean #, ls_02, ls_98
            else :
                return ls_mean



# * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * *#    Task IV: Median NL     * * *  * * # * * * *  * *# # * * * *  * * # * * * *  * * # * * * *  * *#

# function to derive median NL image for the year ()2014)
def med_NLimg(nlpath , year, input_zone_polygon_0, metric='median'):

    ls = []
    for file in glob(os.path.join(nlpath, '*' + year + '*'+ '.tif')):

        # first cut big India from grlobal VIIRS
        imgarray, datamask = srs.zone_mask(input_zone_polygon_0, file)

        #append all images
        ls.append(imgarray)

    arr = np.array(ls)

    if metric == 'median':

        return np.median(arr, axis =0)

    if metric == 'max':
        return np.max(arr, axis=0)

    if metric == 'std':
        return np.std(arr, axis=0)

    if metric == 'p90':
        return np.percentile(arr, 90, axis=0)

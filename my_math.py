# -*- coding: utf-8 -*-
#!/usr/bin/env python
# __author__ = 'Prakhar'
# Created 7/26/2016
# Last edit 7/30/2016

#Purpose : To store all mathematical analysis operation performed e.g. thresholding, etc. for the purpose of being used in another programs

import numpy
import matplotlib.pyplot as plt
import infoFinder as info
import shp_rstr_stat as srs
import numpy as np


# 1. Rosin Thresholding

def RosinThreshold(img, MINVAL ):

    # for unimodal images use maximum deviation algorithm as developed by Rosin
    # adapted from http://www.mathworks.com/matlabcentral/fileexchange/45443-rosin-thresholding/content/RosinThreshold.m
    # Implementation of Rosin Thresholding.
    # Compute the Rosin threshold for an image
    # Takes histogram of an image filtered by any edge detector as as input
    # and return the index which corresponds to the threshold in histogram
    #
    # REF: "Unimodal thresholding" by Paul L. Rosin (2001)
    # "The proposed bilevel thresholding algorithm is ex-tremely simple. It assumes that there is one dominant
    # population in the image that produces one main peak located at the lower end of the histogram relative to the
    # secondary population. This latter class may or may not produce a discernible peak, but needs to be reasonably
    # well separated from the large peak to avoid being swamped by it."

    # MINVAL is the minimum you wish to provide for the hostogram to prevent histogram from being skewed towards zero. for MODIS it is 50, for SO2 10, no2, 10, DNB, 30
    g2 = (img[img > MINVAL]).tolist()
    # low values are removed. they are hugein number mainly opresent in forest, mountains and can skew our threshold towards darker regions than brighter.
    # our aim is to identify threshold between bright and brighter regions as they

    #using Freedman-Diaconis  rule to calculate bin width h=2∗IQR∗n−1/3. So the number of bins is (max-min)/h.
    #http://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram-for-n-where-n-ranges-from-30
    q75, q25 = numpy.percentile(g2, [75, 25])
    iqr = q75 - q25 # interquartile range
    num_bin = int((np.nanmax(g2) - np.nanmin(g2))/(2*iqr/(np.size(g2))**(1.0/3.0)))
    num_bin = 32
    fig = plt.figure()
    hisarr=plt.hist(g2, bins=num_bin)

    peak_max = max(hisarr[0])
    pos_peak = int(hisarr[1][numpy.where(hisarr[0]==peak_max)][0])
    p1 = [pos_peak, peak_max]
    ind_nonZero = hisarr[0][-1]
    last_zeroBin = hisarr[1][-1]
    p2 = [last_zeroBin, ind_nonZero]
    m = (p2[1]-p1[1])/(p2[0]-p1[0])
    c = p2[1] - m*p2[0]
    distmax = 0
    for ind in range(np.nonzero(hisarr[0] == peak_max)[0][0]+1, np.size(hisarr[0])):
        dist = abs((m*hisarr[1][ind] -1*hisarr[0][ind] + c)/m)
        if dist>distmax:
          distmax=dist
          indmax = ind

    print "threshold is ",hisarr[1][indmax]
    thresh = hisarr[1][indmax]
    return thresh
# function end
#------------------------------------------------------



# 2. for bimodal images use Otsu technique
def Otsu_thresh(img):
    import mahotas as mh
    T_otsu = mh.thresholding.otsu(ceil(img).astype(uint8))
    return T_otsu
# function end
# ------------------------------------------------------



# 3. Mean annual image
# to find mean annual images from a deries of daily levle images
def mean_annual_img(raster_path, sat, prod,input_zone_polygon, georef_raster, start,end):
    filenamelist = info.filenamefinder(raster_path, sat, prod)
    sum0=empty(shape(srs.zone_mask(input_zone_polygon, filenamelist[0])[0]))
    mask = srs.zone_mask(input_zone_polygon, filenamelist[0])[1]
    for year in range (start,end):
        sum = sum0
        i=0
        for input_value_raster in filenamelist:
            yearm = info.yearfinder(input_value_raster, sat)
            print yearm
            if int(int(yearm)/100) == year:
                print 'true'
                sum = sum + srs.zone_mask(input_zone_polygon, input_value_raster)[0]
                i = i+1
        avg = (sum/i)*mask # to get rid of masked value we multiply by mask
        srs.arr_to_raster(avg, georef_raster, sat +prod+ str(year) + ".tif")
# function end



# 4. Classification AQ
#perform simple classification for a pair of AQ and DNB image is the thresholds and info about urban cover dummy is specified
def AQ_class(result2, sat_data, YEARM, aq_thresh, prodS, urban):
    # 1 LLHP, 2 HLHP, 3 LLLP, 4 HLLP
    YEARMT = YEARM + '00'
    YEARM = YEARM + '00'
    result2[sat_data[1] + 'filter'] = 0
    result2.ix[(result2[sat_data[1] + YEARMT] >= aq_thresh[sat_data[1]]) & (result2[prodS + YEARM] < aq_thresh[prodS]), sat_data[1] + 'filter'] = 1
    result2.ix[(result2[sat_data[1] + YEARMT] >= aq_thresh[sat_data[1]]) & (result2[prodS + YEARM] >= aq_thresh[prodS]), sat_data[1] + 'filter'] = 2
    result2.ix[(result2[sat_data[1] + YEARMT] < aq_thresh[sat_data[1]]) & (result2[prodS + YEARM] < aq_thresh[prodS]), sat_data[1] + 'filter'] = 3
    result2.ix[(result2[sat_data[1] + YEARMT] < aq_thresh[sat_data[1]]) & (result2[prodS + YEARM] >= aq_thresh[prodS]), sat_data[1] + 'filter'] = 4

    cls_all = [len(result2[result2[sat_data[1] + 'filter'] == 1]),
               len(result2[result2[sat_data[1] + 'filter'] == 2]),
               len(result2[result2[sat_data[1] + 'filter'] == 3]),
               len(result2[result2[sat_data[1] + 'filter'] == 4])]

    cls_urban = [len(result2[(result2[sat_data[1] + 'filter'] == 1) & (result2['LULC'] == 1)]),
                 len(result2[(result2[sat_data[1] + 'filter'] == 2) & (result2['LULC'] == 1)]),
                 len(result2[(result2[sat_data[1] + 'filter'] == 3) & (result2['LULC'] == 1)]),
                 len(result2[(result2[sat_data[1] + 'filter'] == 4) & (result2['LULC'] == 1)])]

    if urban == 0:
        print sat_data[1] + YEARM, ', ', '\t '.join(str(p) for p in cls_all)
    if urban == 1:
        print sat_data[1] + YEARM, ', ', '\t '.join(str(p) for p in cls_urban)
    return result2, cls_all, cls_urban
#function end

# RUN FUNCTION AQ_class


# 5. Clount number of non-zero along an axis. currently only for 3D and for axis 0
def count_nonzero(arr, axis):

    # get shape of input
    shape = arr[0].shape

    # create an ampty array
    ls = np.zeros(shape)

    # check for each pitwell how many non zeros
    for i in range(arr[0].shape[0]):
        for j in range(arr[0].shape[1]):

           # this numpy function will be upgraded with numpy and may chaneg behaviour
           ls[i,j] = np.count_nonzero(arr[:,i,j])

    return ls

# 6. Funciotns to fit trend line
# functions to determine trendline
def funclog(x, a, b):
    return a * np.log(x) + b
def funclin(x, a, b):
    return a * x + b







# code from https://pythonprogramming.net/3d-graphing-pandas-matplotlib/
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
import seaborn as sns




# df = pd.read_csv('E:\Acads\Research\AQM research\Docs prepared\Excel files\Shp_mean_AOD_DNB_ANG.csv', header=0 ,parse_dates=True)
# print(df.head())
# df = pd.read_csv('E:\Acads\Research\AQM research\Docs prepared\Excel files\Shp_mean_AOD_DNB_ANG.csv', header=0 ,parse_dates=True)
# Source on great plots http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
# http://blog.olgabotvinnik.com/prettyplotlib/;
#most usable perhaps C:\Users\Prakhar\Anaconda2\Scripts


# --------------------------------------------------------------   Plots 3D plots    --------------------------------------------------------------
#function to plot 3D plots from dataframe
def dfPlot3D(df, Xaxis, Yaxis, Zaxis):
    # df['H-L'] = df.High - df.Low
    # df['100MA'] = pd.rolling_mean(df['Close'], 100)

    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(df[Xaxis],  df[Yaxis], df[Zaxis])
    threedee.set_xlabel(Xaxis)
    threedee.set_ylabel(Yaxis)
    threedee.set_zlabel(Zaxis)
    plt.show()


# --------------------------------------------------------------     plot array as an equalized histogram image   --------------------------------------------------------------
def histeq(im,nbr_bins=256):
    """ function to display array as images by equalizing them from 0-255


    Parameters
    ----------
    im : array name
    nbr_bins :

   #get image histogram
   """
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)
    fig=plt.figure()
    plt.imshow(im2.reshape(im.shape))
    # return im2.reshape(im.shape), cdf



#--------------------------------------------------------------    Plot 2D scatter      --------------------------------------------------------------
#function toplot 2d graphs from data using 2 dataframes
def SctPlot2D(Xaxis, Yaxis, subt, Xlabel, Ylabel, save_path, save=0):

    title_font = {'fontname': 'Arial', 'size': '20', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '16'}
    fig=plt.figure()
    plt.scatter(Xaxis,Yaxis)
    fig.suptitle(subt, **axis_font)
    plt.xlabel(Xlabel, **axis_font)
    plt.ylabel(Ylabel, **title_font)
    if save==1:
        fig.savefig(save_path)


#--------------------------------------------------------------    Plot with basemap       --------------------------------------------------------------

#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as plt
#import seaborn
#map = Basemap(projection='ortho',
#              lat_0=8, lon_0=70)
#
##Fill the globe with a blue color
#map.drawmapboundary(fill_color='aqua')
##Fill the continents with the land color
## map.fillcontinents(color='coral',lake_color='aqua')
#
#map.drawcoastlines()
#map.bluemarble()
#plt.show()

#--------------------------------------------------------------    Plot desne scatter       --------------------------------------------------------------
# https://pypi.python.org/pypi/mpl-scatter-density
import numpy as np
import mpl_scatter_density
import matplotlib.pyplot as plt

# Generate fake data

N = 10000000
x = np.random.normal(4, 2, N)
y = np.random.normal(3, 1, N)

# Make the plot - note that for the projection option to work, the
# mpl_scatter_density module has to be imported above.

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
ax.scatter_density(x, y)
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)


N = 10000000
x = np.random.normal(4, 2, N)
y = np.random.normal(3, 1, N)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
density = ax.scatter_density(x, y)
ax.set_xlim(-5, 10)
ax.set_ylim(-5, 10)
fig.colorbar(density, label='Number of points per pixel')


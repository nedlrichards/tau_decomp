import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import matplotlib.tri as tri
from scipy.ndimage import gaussian_filter

from src import Config

plt.ion()

cf=Config()

woa_file = np.genfromtxt('data/external/woa18_decav81B0_t14mn04.csv',
                         delimiter=',', missing_values='',
                         filling_values=np.nan,
                         usecols=(0,1,12), invalid_raise=False).T

xi = np.linspace(-160, -115, 100)
yi = np.linspace(15, 50, 101)

lat_exp = [33.42, 34.88]
lon_exp = [-137.7, -148.32]

ind1 = (woa_file[0] > 15) & (woa_file[0] < 50)
ind2 = (woa_file[1] > -160) & (woa_file[1] < -115)
ind = ind1 & ind2
nan_i = ~np.isnan(woa_file[2])
ind &= nan_i

triang = tri.Triangulation(woa_file[1, ind], woa_file[0, ind])
interpolator = tri.LinearTriInterpolator(triang, woa_file[2, ind])


fig = plt.figure(figsize=(cf.jasa_1clm, 2.5))
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
ax.set_extent([-160, -115, 15, 50],crs=ccrs.PlateCarree())
ax.coastlines()

ax.plot(lon_exp, lat_exp, color='C3')
#m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')

Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)
data = gaussian_filter(zi, 1)

cs = ax.contour(xi, yi, data, linewidths=0.5, colors='k', levels=np.arange(6, 26, 2))
locs = [(-155.4545454545455, 47.200126321991945),
        (-149.54545454545456, 44.78292593479905),
        (-135.45454545454547, 43.84985763955887),
        (-129.09090909090912, 39.75988868530165),
        (-135, 36.89),
        (-132.1, 33.66),
        (-135., 30.5),
        (-139.5, 27.35),
        (-146.4, 24.35),
        (-151.5509215051357, 20.25)]
lbls = ax.clabel(cs, cs.levels, manual=locs)

ax.stock_img()

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
#parallels = np.linspace(20, 50, 5)
#m.drawparallels(parallels,labels=[False,True,True,False])

#meridians = np.linspace(-115, -155, 5)
#m.drawmeridians(meridians,labels=[True,False,False,True])

fig.savefig('reports/jasa/figures/transcet.png', dpi=300)

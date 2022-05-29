import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.tri as tri

plt.ion()

woa_file = np.genfromtxt('data/external/woa18_95A4_t13mn04.csv', delimiter=',',
                         missing_values='', filling_values=np.nan,
                         usecols=(0,1,22), invalid_raise=False).T
lats = [33.42, 34.88]
lons = [-137.7, -148.32]

fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
ax.set_extent([-160, -115, 15, 50],crs=ccrs.PlateCarree())
ax.coastlines()
#m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')

xi = np.linspace(-160, -115, 100)
yi = np.linspace(15, 50, 101)
ax.plot(lons, lats, color='C3')
ind1 = (woa_file[0] > 15) & (woa_file[0] < 50)
ind2 = (woa_file[1] > -160) & (woa_file[1] < -115)
ind = ind1 & ind2
nan_i = ~np.isnan(woa_file[2])
ind &= nan_i

triang = tri.Triangulation(woa_file[1, ind], woa_file[0, ind])
interpolator = tri.LinearTriInterpolator(triang, woa_file[2, ind])

from scipy.ndimage import gaussian_filter

Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)
data = gaussian_filter(zi, 1)
cs = ax.contour(xi, yi, data, linewidths=0.5, colors='k', levels=np.arange(6, 26, 2))
ax.clabel(cs, cs.levels)

ax.stock_img()

gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
ax.set_title('100 m isotherm contours')
#parallels = np.linspace(20, 50, 5)
#m.drawparallels(parallels,labels=[False,True,True,False])

#meridians = np.linspace(-115, -155, 5)
#m.drawmeridians(meridians,labels=[True,False,False,True])

fig.savefig('reports/jasa/figures/transcet.png', dpi=300)

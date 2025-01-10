import numpy as np
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from siphon.catalog import TDSCatalog
import xarray as xr
from scipy.ndimage import gaussian_filter
from metpy.units import units
import time

script_start = time.time()

print('---------------------------------------')
print('GFS Vorticity Script - Script started.')
print('---------------------------------------')

def find_time_dim(ds, var_name):
    possible_time_dims = ['time', 'time1', 'time2', 'time3']
    time_dim = None
    for dim in possible_time_dims:
        if dim in ds[var_name].dims:
            time_dim = dim
            break
    if time_dim is None:
        raise ValueError('Could not find the time dimension')
    return time_dim
    
def find_press_dim(ds, var_name):
    possible_iso_dims = ['isobaric', 'isobaric1', 'isobaric2', 'isobaric3']
    iso_dim = None
    for dim in possible_iso_dims:
        if dim in ds[var_name].dims:
            iso_dim = dim
            break
    if iso_dim is None:
        raise ValueError('Could not find the iso dimension')
    return iso_dim

# Helper function for finding proper time variable
def find_time_var(var, time_basename='time'):
    for coord_name in var.coords:
        if coord_name.startswith(time_basename):
            return var.coords[coord_name]
    raise ValueError('No time variable found for ' + var.name)

def find_press_var(var, time_basename='isobaric'):
    for coord_name in var.coords:
        if coord_name.startswith(time_basename):
            return var.coords[coord_name]
    raise ValueError('No time variable found for ' + var.name)

tds_gfs = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.html')
gfs_ds = tds_gfs.datasets[0]
ds = xr.open_dataset(gfs_ds.access_urls['OPENDAP'])
ds_latlon = ds.sel(lat=slice(60, 15), lon=slice(360-140, 360-50))

dims_to_keep = ['lat', 'lon', 'time', 'time1', 'time2', 'time3', 'isobaric']

ds_latlon = ds_latlon.drop_dims([dim for dim in ds_latlon.dims if dim not in dims_to_keep])

time_dim = find_time_var(ds_latlon['Temperature_isobaric'])
iso_dim = find_press_var(ds_latlon['Temperature_isobaric'])

init_time = time_dim[0].values

target_length = len(time_dim)  

# Initialize a variable to store the matching dimension name
matching_dim = None

# Loop over the dimensions and check their lengths
for dim, size in ds_latlon.dims.items():
    if size == target_length:
        matching_dim = dim
        break  # Exit loop once a match is found

for i in range(0, 29, 2):
    iteration_start = time.time()
    ds = ds_latlon.isel(**{matching_dim: i})

    # Extract the variables
    gph_500  = ds['Geopotential_height_isobaric'].sel(isobaric=50000) * units.meter
    uwnd_500 = ds['u-component_of_wind_isobaric'].sel(isobaric=50000) * units.meter / units.second
    vwnd_500 = ds['v-component_of_wind_isobaric'].sel(isobaric=50000) * units.meter / units.second
    t_850 = (ds['Temperature_isobaric'].sel(isobaric=85000) - 273.15) * units.celsius
    uwnd_850 = ds['u-component_of_wind_isobaric'].sel(isobaric=85000) * units.meter / units.second
    vwnd_850 = ds['v-component_of_wind_isobaric'].sel(isobaric=85000) * units.meter / units.second

    absolute_vorticity = mpcalc.absolute_vorticity(uwnd_500, vwnd_500) # units 1/s
    tadv_850 = mpcalc.advection(t_850, uwnd_850, vwnd_850) * 3600 # units: K/hr

    lons, lats = ds['lon'], ds['lat']

    gph_500_smoothed = gaussian_filter(gph_500, sigma=3.0)
    absolute_vorticity_smoothed = gaussian_filter(absolute_vorticity, sigma=3.0)
    tadv_850_smoothed = gaussian_filter(tadv_850, sigma=3.0)

    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.LambertConformal()})
    ax.set_extent([-125, -66.9, 23, 49.4])
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    # Plot the isohypses, absolute vorticity, and temperature advection
    isohypses = plt.contour(lons, lats, gph_500_smoothed, colors='black', levels=np.arange(5000, 6200, 60), linewidths=1.25, transform=ccrs.PlateCarree())
    try:
        plt.clabel(isohypses, fontsize=10, inline=1, inline_spacing=3, fmt='%d', rightside_up=True, use_clabeltext=True)
    except IndexError:
        print("No contours to label for isohypses.")

    vort_cf = plt.contourf(lons, lats, absolute_vorticity_smoothed * 1e5, cmap='plasma_r', levels=np.arange(10,65,5), transform=ccrs.PlateCarree(), extend='max')

    cadv = plt.contour(lons, lats, tadv_850_smoothed, levels=np.arange(-10, -1, 0.5), colors='blue', linewidths=0.75, linestyles='solid', transform=ccrs.PlateCarree())
    try:
        plt.clabel(cadv, fontsize=10, inline=1, inline_spacing=3, fmt='%.1f', rightside_up=True, use_clabeltext=True)
    except IndexError:
        print("No contours to label for cadv.")
    wadv = ax.contour(lons, lats, tadv_850_smoothed, levels=np.arange(1, 10, 0.5), colors='red', linewidths=0.75, linestyles='solid', transform=ccrs.PlateCarree())
    try:
        plt.clabel(wadv, fontsize=10, inline=1, inline_spacing=3, fmt='%.1f', rightside_up=True, use_clabeltext=True)
    except IndexError:
        print("No contours to label for wadv.")

    # Adding the legend
    isohypses_line = plt.Line2D([0], [0], color='black', linewidth=1, label='Geopotential Height (m)')
    cadv_line = plt.Line2D([0], [0], color='blue', linewidth=1, label='Cold Advection (K/hr)')
    wadv_line = plt.Line2D([0], [0], color='red', linewidth=1, label='Warm Advection (K/hr)')
    ax.legend(handles=[isohypses_line, cadv_line, wadv_line], loc='upper right')

    hour_difference = (ds_latlon[matching_dim][i] - init_time) / np.timedelta64(1, 'h')

    plt.title(f"{ds_latlon[matching_dim][0].dt.strftime('%H00 UTC').item()} GFS 500-hPa Geopotential Heights, Absolute Vorticity, and 850-hPa Temperature Advection | {ds_latlon[matching_dim][i].dt.strftime('%Y-%m-%d %H00 UTC').item()} | FH: {hour_difference:.0f}", fontsize=12)
    plt.colorbar(vort_cf, orientation='horizontal', label='Absolute Vorticity ($10^{-5}$ s$^{-1}$)', pad=0.05, aspect=50)
    plt.tight_layout()
    plt.savefig(f'gfs/vort/{hour_difference:.0f}.png', dpi=450, bbox_inches='tight')
    iteration_end = time.time()
    print(f'Iteration {i} Processing Time:', round((iteration_end - iteration_start), 2), 'seconds.')

print('\nTotal Processing Time:', round((time.time() - script_start), 2), 'seconds.')

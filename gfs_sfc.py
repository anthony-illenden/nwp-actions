import numpy as np
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from siphon.catalog import TDSCatalog
import xarray as xr
from scipy.ndimage import gaussian_filter
from metpy.units import units
import metpy

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

for i in range(0, 29):
    ds = ds_latlon.isel(**{matching_dim: i})

    # Extract the variables
    gph_500  = ds['Geopotential_height_isobaric'].sel(isobaric=50000) * units.meter
    gph_1000 = ds['Geopotential_height_isobaric'].sel(isobaric=100000) * units.meter
    dbz = ds['Composite_reflectivity_entire_atmosphere']
    mslp = (ds['MSLP_Eta_model_reduction_msl'] / 100) * units.hPa  # Ensure mslp is 2D

    thickness = ((gph_500 - gph_1000) / 10) * units.decameter

    thickness_smoothed = gaussian_filter(thickness, sigma=3.0)
    mslp_smoothed = gaussian_filter(mslp, sigma=3.0)

    lons, lats = ds['lon'], ds['lat']

    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.LambertConformal()})
    ax.set_extent([-125, -66.9, 23, 49.4])
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    isobars = plt.contour(lons, lats, mslp_smoothed, colors='black', levels=np.arange(940, 1080, 4), linewidths=1, transform=ccrs.PlateCarree())
    try:
        plt.clabel(isobars, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
    except IndexError:
        print("No contours to label for isobars.")

    thickness_546 = plt.contour(lons, lats, thickness_smoothed, colors='red', levels=np.arange(546, 600, 6), linestyles='dashed', linewidths=0.75, transform=ccrs.PlateCarree())
    try:
        plt.clabel(thickness_546, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
    except IndexError:
        print("No contours to label for thickness.")
    thickness_540 = plt.contour(lons, lats, thickness_smoothed, colors='blue', levels=np.arange(420, 546, 6), linestyles='dashed', linewidths=0.75, transform=ccrs.PlateCarree())
    try:
        plt.clabel(thickness_540, inline=True, inline_spacing=5, fontsize=10, fmt='%i')
    except IndexError:
        print("No contours to label for thickness.")

    dbz_cf = plt.contourf(lons, lats, dbz, levels=np.arange(5, 80, 5), cmap=metpy.plots.ctables.registry.get_colortable('NWSReflectivity'), transform=ccrs.PlateCarree())

    hour_difference = (ds['time'][i] - init_time) / np.timedelta64(1, 'h')

    # Adding the legend
    isobar_lines = plt.Line2D([0], [0], color='black', linewidth=1, label='MSLP (hPa)')
    cadv_line = plt.Line2D([0], [0], color='blue', linestyle='dashed', linewidth=1, label='Thickness <=540 (dam)')
    wadv_line = plt.Line2D([0], [0], color='red', linestyle='dashed', linewidth=1, label='Thickness >540 (dam)')
    ax.legend(handles=[isobar_lines, cadv_line, wadv_line], loc='upper right')

    plt.title(f"{ds_latlon[matching_dim][0].dt.strftime('%H00 UTC').item()} GFS 100-500 hPa Thickness, MSLP, and Reflectivity | {ds[matching_dim].dt.strftime('%Y-%m-%d %H00 UTC').item()} | FH: {hour_difference:.0f}", fontsize=12)
    plt.colorbar(dbz_cf, orientation='horizontal', label='Reflectivity (dBZ)', pad=0.05, aspect=50)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'gfs/sfc/{hour_difference:.0f}.png')
    plt.close()

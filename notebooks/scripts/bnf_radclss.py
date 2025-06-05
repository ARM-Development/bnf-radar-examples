import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import glob
import time
import datetime
import argparse
import logging
import dask

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from dask.distributed import Client, LocalCluster, wait, as_completed, fire_and_forget
dask.config.set({'logging.distributed': 'error'})

from matplotlib.dates import DateFormatter
from matplotlib import colors

import pyart
import act

#-----------------
# Define Functions
#-----------------

def subset_points(nfile, **kwargs):
    """
    Subset a radar file for a set of latitudes and longitudes
    utilizing Py-ART's column-vertical-profile functionality.

    Parameters
    ----------
    file : str
        Path to the radar file to extract columns from
    nsonde : list
        List containing file paths to the desired sonde file to merge

    Calls
    -----
    radar_start_time
    merge_sonde

    Returns
    -------
    ds : xarray DataSet
        Xarray Dataset containing the radar column above a give set of locations
    
    """
    ds = None
    
    # Define the splash locations [lon,lat]
    M1 = [34.34525, -87.33842]
    S4 = [34.46451,	-87.23598]
    S20 = [34.65401, -87.29264]
    S30	= [34.38501, -86.92757]
    S40	= [34.17932, -87.45349]

    sites = ["M1", "S4", "S20", "S30", "S40"]

    # Zip these together!
    lats, lons = list(zip(M1,
                          S4,
                          S20,
                          S30,
                          S40))
    try:
        # Read in the file
        radar = pyart.io.read(nfile)
        # Check for single sweep scans
        if np.ma.is_masked(radar.sweep_start_ray_index["data"][1:]):
            radar.sweep_start_ray_index["data"] = np.ma.array([0])
            radar.sweep_end_ray_index["data"] = np.ma.array([radar.nrays])
    except:
        radar = None

    if radar:
        if radar.scan_type != "rhi":
            if radar.time['data'].size > 0:
                # Easier to map the nearest sonde file to radar gates before extraction
                if 'sonde' in kwargs:
                    # variables to discard when reading in the sonde file
                    exclude_sonde = ['base_time', 'time_offset', 'lat', 'lon', 'qc_pres',
                                    'qc_tdry', 'qc_dp', 'qc_wspd', 'qc_deg', 'qc_rh',
                                    'qc_u_wind', 'qc_v_wind', 'qc_asc']
        
                    # find the nearest sonde file to the radar start time
                    radar_start = datetime.datetime.strptime(nfile.split('/')[-1].split('.')[-3] + '.' + nfile.split('/')[-1].split('.')[-2], 
                                                            '%Y%m%d.%H%M%S'
                    )
                    sonde_start = [datetime.datetime.strptime(xfile.split('/')[-1].split('.')[2] + 
                                                              '-' + 
                                                              xfile.split('/')[-1].split('.')[3], 
                                                              '%Y%m%d-%H%M%S') for xfile in kwargs['sonde']
                    ]
                    # difference in time between radar file and each sonde file
                    start_diff = [radar_start - sonde for sonde in sonde_start]

                    # merge the sonde file into the radar object
                    ds_sonde = act.io.read_arm_netcdf(kwargs['sonde'][start_diff.index(min(start_diff))], 
                                                      cleanup_qc=True, 
                                                      drop_variables=exclude_sonde)
   
                    # create list of variables within sonde dataset to add to the radar file
                    for var in list(ds_sonde.keys()):
                        if var != "alt":
                            z_dict, sonde_dict = pyart.retrieve.map_profile_to_gates(ds_sonde.variables[var],
                                                                                    ds_sonde.variables['alt'],
                                                                                    radar)
                        # add the field to the radar file
                        radar.add_field_like('corrected_reflectivity', "sonde_" + var,  sonde_dict['data'], replace_existing=True)
                        radar.fields["sonde_" + var]["units"] = sonde_dict["units"]
                        radar.fields["sonde_" + var]["long_name"] = sonde_dict["long_name"]
                        radar.fields["sonde_" + var]["standard_name"] = sonde_dict["standard_name"]
                        radar.fields["sonde_" + var]["datastream"] = ds_sonde.datastream

                    del radar_start, sonde_start, ds_sonde
                    del z_dict, sonde_dict
        
                column_list = []
                for lat, lon in zip(lats, lons):
                    # Make sure we are interpolating from the radar's location above sea level
                    # NOTE: interpolating throughout Troposphere to match sonde to in the future
                    try:
                        da = (
                            pyart.util.columnsect.column_vertical_profile(radar, lat, lon)
                            .interp(height=np.arange(500, 8500, 250))
                        )
                    except ValueError:
                        da = pyart.util.columnsect.column_vertical_profile(radar, lat, lon)
                        # drop the NaNs (and associated fields) from the extraction
                        da = da.isel(height=~np.isnan(da.height))
                        # interpolate to the same profile 
                        da = da.interp(height=np.arange(500, 8500, 250))

                    # Interpolate NaNs out
                    da = da.interpolate_na(dim="height", method="linear", fill_value="extrapolate")   
                    # Add the latitude and longitude of the extracted column
                    da["latitude"], da["longitude"] = lat, lon
                    # Convert timeoffsets to timedelta object and precision on datetime64
                    da.time_offset.data = da.time_offset.values.astype("timedelta64[s]")
                    da.base_time.data = da.base_time.values.astype("datetime64[s]")
                    # Time is based off the start of the radar volume
                    da["gate_time"] = da.base_time.values + da.isel(height=0).time_offset.values
                    column_list.append(da)
        
                # Concatenate the extracted radar columns for this scan across all sites    
                ds = xr.concat([data for data in column_list if data], dim='station')
                ds["station"] = sites
                # Add attributes for Time, Latitude, Longitude, and Sites
                ds.gate_time.attrs.update(long_name=('Time in Seconds that Cooresponds to the Start'
                                                    + " of each Individual Radar Volume Scan before"
                                                    + " Concatenation"),
                                          description=('Time in Seconds that Cooresponds to the Minimum'
                                                    + ' Height Gate'))
                ds.time_offset.attrs.update(long_name=("Time in Seconds Since Midnight"),
                                            description=("Time in Seconds Since Midnight that Cooresponds"
                                                        + "to the Center of Each Height Gate"
                                                        + "Above the Target Location ")
                                            )
                ds.station.attrs.update(long_name="Bankhead National Forest AMF-3 In-Situ Ground Observation Station Identifers")
                ds.latitude.attrs.update(long_name='Latitude of BNF AMF-3 Ground Observation Site',
                                         units='Degrees North')
                ds.longitude.attrs.update(long_name='Longitude of BNF AMF-3 Ground Observation Site',
                                          units='Degrees East')
                # delete the radar to free up memory
                del radar, column_list, da
            else:
                # delete the rhi file
                del radar
        else:
            del radar
    return ds

def match_datasets_act(column, ground, site, discard, resample='sum', DataSet=False):
    """
    Time synchronization of a Ground Instrumentation Dataset to 
    a Radar Column for Specific Locations using the ARM ACT package
    
    Parameters
    ----------
    column : Xarray DataSet
        Xarray DataSet containing the extracted radar column above multiple locations.
        Dimensions should include Time, Height, Site
             
    ground : str; Xarray DataSet
        String containing the path of the ground instrumentation file that is desired
        to be included within the extracted radar column dataset. 
        If DataSet is set to True, ground is Xarray Dataset and will skip I/O. 
             
    site : str
        Location of the ground instrument. Should be included within the filename. 
        
    discard : list
        List containing the desired input ground instrumentation variables to be 
        removed from the xarray DataSet. 
    
    resample : str
        Mathematical operational for resampling ground instrumentation to the radar time.
        Default is to sum the data across the resampling period. Checks for 'mean' or 
        to 'skip' altogether. 
    
    DataSet : boolean
        Boolean flag to determine if ground input is an Xarray Dataset.
        Set to True if ground input is Xarray DataSet. 
             
    Returns
    -------
    ds : Xarray DataSet
        Xarray Dataset containing the time-synced in-situ ground observations with
        the inputed radar column 
    """
    # Check to see if input is xarray DataSet or a file path
    if DataSet == True:
        grd_ds = ground
    else:
        # Read in the file using ACT
        grd_ds = act.io.read_arm_netcdf(ground, cleanup_qc=True, drop_variables=discard)
        # Default are Lazy Arrays; convert for matching with column
        grd_ds = grd_ds.compute()
        
    # Remove Base_Time before Resampling Data since you can't force 1 datapoint to 5 min sum
    if 'base_time' in grd_ds.data_vars:
        del grd_ds['base_time']
        
    # Check to see if height is a dimension within the ground instrumentation. 
    # If so, first interpolate heights to match radar, before interpolating time.
    if 'height' in grd_ds.dims:
        grd_ds = grd_ds.interp(height=np.arange(500, 8500, 250), method='linear')
        
    # Resample the ground data to 5 min and interpolate to the CSAPR-2. 
    # Keep data variable attributes to help distingish between instruments/locations
    if resample.split('=')[-1] == 'mean':
        matched = grd_ds.resample(time='5Min', 
                                  closed='right').mean(keep_attrs=True).interp(time=column.time, 
                                                                               method='linear')
    elif resample.split('=')[-1] == 'skip':
        matched = grd_ds.interp(time=column.time, method='linear')
    else:
        matched = grd_ds.resample(time='5Min', 
                                  closed='right').sum(keep_attrs=True).interp(time=column.time, 
                                                                              method='linear')
    
    # Add BNF site location as a dimension for the Pluvio data
    matched = matched.assign_coords(coords=dict(station=site))
    matched = matched.expand_dims('station')
   
    # Remove Lat/Lon Data variables as it is included within the Matched Dataset with Site Identfiers
    if 'lat' in matched.data_vars:
        del matched['lat']
    if 'lon' in matched.data_vars:
        del matched['lon']
    if 'alt' in matched.data_vars:
        del matched['alt']
        
    # Update the individual Variables to Hold Global Attributes
    # global attributes will be lost on merging into the matched dataset.
    # Need to keep as many references and descriptors as possible
    for var in matched.data_vars:
        matched[var].attrs.update(source=matched.datastream)
        
    # Merge the two DataSets
    column = xr.merge([column, matched])
   
    return column

def adjust_dod(ds, ntime, height_fix=False):
    """
    the ability to create a DOD with adjustable dimensions via ACT is not 
    allowed on specific nodes

    therefore, using the stored DOD file for RadCLss, adjust the time 
    dimension as needed. 

    Input
    -----
    ds : xarray Dataset
        The input DOD to have the time variable adjusted
    
    ntime : int
        New dimension to expand or shrink the DOD time dimension to

    Output
    ------
    adj_ds : xarray Dataset
        blank dataset containing the DOD metadata adjusted for proper time 
        dimensions
    """
    # Create a blank DataSet
    newds = xr.Dataset()
    
    # Get the global attributes and add to dataset
    newds.attrs = ds.attrs
    if height_fix is True:
        nskip = ['latitude', 'longitude', 'base_time']
        nheight = np.arange(3150, 10050, 50)
    else:
        nskip = ['lat', 'lon']

    # Assign the variables to the DataSet, expand the blank arrays to the input int time
    for var in ds.data_vars:
        if var not in nskip:
            if height_fix is True:
                x = np.full(len(nheight), ds[var].data[0])
                newds[var] = ('height', x)
                newds[var].attrs = ds[var].attrs
            else:
                if len(ds[var].data.shape) == 4:
                    x = np.full((ntime, ds[var].data.shape[1],
                                 ds[var].data.shape[2],
                                 ds[var].data.shape[3]),
                                 ds[var].data[0, 0, 0, 0])
                elif len(ds[var].data.shape) == 3:
                    x = np.full((ntime, ds[var].data.shape[1],
                                 ds[var].data.shape[2]), ds[var].data[0, 0, 0])
                elif len(ds[var].data.shape) == 2:
                    x = np.full((ntime, ds[var].data.shape[1]), ds[var].data[0, 0])
                else:
                    x = np.full((ntime), ds[var].data[0])
                newds[var] = (ds[var].dims, x)
                newds[var].attrs = ds[var].attrs
    if height_fix is True:
        newds['latitude'] = ds['latitude']
        newds['longitude'] = ds['longitude']
    else:
        # Skipped the variables without time, add those back in
        newds['lat'] = ds['lat']
        newds['lon'] = ds['lon']
    
    if height_fix is True:
        newds = newds.assign_coords(height=nheight)
    else:
        # Assign Coordinates to the array
        newds = newds.assign_coords(time=np.arange(0, newds['time'].shape[0]),
                                    height=np.arange(0, newds['height'].shape[0]),
                                    station=np.arange(0, 6),
                                    particle_size=np.arange(0, 32),
                                    raw_fall_velocity=np.arange(0, 32)
                                   )
    
    return newds

def create_radclss_figure(radclss, height=3500, outdir=None):
    """
    With the RadCLss product, generate a timeseries of radar reflectivity factor, particle size distribution and cumuluative precipitaiton 
    for the ARM SAIL M1 Site. 

    This timeseries quick is to serve as a means for evaluating the RadCLss product.

    Input
    -----
    nfile : str
        Filepath to the RadCLss file.
    height : int
        Column height to compare against in-situ sensors for precipitation accumulation. 
    outdir : str
        Path to desired output directory. If not supplied, assumes current working directory.

    Output
    ------
    timeseries : png
        Saved image of the RadCLss timeseris
    """
    # Define a status
    status = None
    # Define the date of the file
    DATE = radclss.time.data[0].astype(str).split('T')[0].replace('-', '')
    
    # Calculate the daily accumulated precipitation for the laser disdrometer and weighing bucket 
    try:
        ld_accum = act.utils.accumulate_precip(radclss.sel(station="M1"), "precip_rate").precip_rate_accumulated.compute()
    except:
        ld_accum = None
    try:
        pluvio_accum = act.utils.accumulate_precip(radclss.sel(station="M1"), "intensity_rtnrt").intensity_rtnrt_accumulated.compute()
    except:
        pluvio_accum = None

    # Resample RadCLss to 1-min temporal frequency for precipiation accumulation calculations
    # Assume min height is 3500 meters
    ds_resampled = radclss.resample(time="1min").mean().sel(height=height).sel(station="M1")
    
    # Define the Z-S relationships used with CMAC-SQUIRE-RadCLss
    zs_fields = {"Wolf_and_Snider": {"A": 110, "B": 2, "name": 'snow_rate_ws2012'},
                 "WSR_88D_Intermountain_West": {"A": 40, "B": 2, "name": 'snow_rate_ws88diw'},
                 "Matrosov et al.(2009) Braham(1990) 1": {"A": 67, "B": 1.28, "name": 'snow_rate_m2009_1'},
                 "Matrosov et al.(2009) Braham(1990) 2": {"A": 114, "B": 1.39, "name": 'snow_rate_m2009_2'},
    }

    # Calculate accumulated precipitation from the Z-S relationships
    for field in zs_fields:
        ds_resampled[zs_fields[field]["name"]].attrs["units"] = "mm/hr"
        ds_resampled = act.utils.accumulate_precip(ds_resampled, zs_fields[field]["name"])

    # Create the figure and subaxes 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(20,12))

    # Define the DateFormatter
    date_form = DateFormatter('%Y-%m-%d \n %H:%M:%S')

    #--------------------------------
    # Plot A - M1 Column Reflectivity
    #--------------------------------
    dbz_plot = radclss.sel(station="M1").corrected_reflectivity.plot(cmap='pyart_HomeyerRainbow',
                                                                     vmin=-20,
                                                                     vmax=40,
                                                                     ax=ax1,
                                                                     add_colorbar=False,
                                                                     figure=fig,
                                                                     y='height')
    ax1.set_ylim(3500, 5000)
    ax1.set_ylabel("Height Above Ground \n (m)", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_title(f'Horizontal Reflectivity at ARM AMF Site', fontsize=20)
    ax1.xaxis.set_major_formatter(date_form)
    ax1.set_xlim(radclss.time.data[0], radclss.time.data[-1])
    ax1.tick_params(axis='both', which='major', labelsize=14)

    #--------------------------------
    # Plot B - Drop Size Distribution
    #--------------------------------

    # Drop Size Distribution
    ds_vmin = np.ma.masked_invalid(radclss.number_density_drops.values).min()+1
    ds_vmax = np.ma.masked_invalid(radclss.number_density_drops.values).max()+2
    if ds_vmax < 0:
        norm = colors.LogNorm(vmin=1,
                              vmax=10)
    else:
        norm = colors.LogNorm(vmin=1,
                              vmax=ds_vmax)

    dsd_plot = radclss.sel(station="M1").number_density_drops.plot(x="time",
                                                                   y="particle_size",
                                                                   norm=norm,
                                                                   cmap="pyart_HomeyerRainbow",
                                                                   add_colorbar=False,
                                                                   ax=ax2,
        )
    ax2.set_ylim(0, 15)
    ax2.set_ylabel("Particle Size \n (mm)", fontsize=14)
    ax2.set_xlabel("")
    ax2.set_title(f'Particle Size Distribution from AMF Laser Disdrometer', fontsize=20)
    ax2.set_xlim(radclss.time.data[0], radclss.time.data[-1])
    ax2.tick_params(axis='both', which='major', labelsize=14)

    #-----------------------------
    # Plot C - Total Accumulation
    #-----------------------------
    for field in zs_fields:
        relationship_name = field.replace("_", " ")
        a_coefficeint = zs_fields[field]["A"]
        b_coefficeint = zs_fields[field]["B"]
        relationship_equation = f"$Z = {a_coefficeint}S^{b_coefficeint}$"
        field_name = zs_fields[field]["name"] + "_accumulated"

        (ds_resampled[field_name]).plot(label=f'{relationship_name} ({relationship_equation})',
                                        ax=ax3
        )
    if ld_accum is not None:
        ld_accum.plot(ax=ax3,label=f"Laser Disdrometer (M1)")
    if pluvio_accum is not None:
        pluvio_accum.plot(ax=ax3,label=f"Pluvio Sensor (M1)")
    ax3.set_title(f"Cumulative Precipitation Comparison", fontsize=20)
    ax3.set_xlim(radclss.time.data[0], radclss.time.data[-1])
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_ylabel("Total Precipitation \n Since 0000 UTC \n (mm)", fontsize=14)
    if pluvio_accum is not None:
        ax3.set_ylim(0, np.max(pluvio_accum)+10)
    else:
        ax3.set_ylim(0, 5)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.xaxis.set_major_formatter(date_form)
    ax3.set_xlabel("Time [UTC]", fontsize=14)

    # ----------
    # Colorbars
    # ----------
    # Reflectivity colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.69, 0.02, 0.165])
    cbar = fig.colorbar(dbz_plot, orientation="vertical", ax=ax1, cax=cbar_ax)
    cbar.set_ticklabels(np.arange(-10, 50, 5), fontsize=14)
    cbar.set_label(label='Horizontal \n Reflectivity \n Factor ($Z_{H}$) \n (dBZ)', fontsize=16)

    # Particle Size Distribution colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax2 = fig.add_axes([0.9, 0.41, 0.02, 0.165])
    cbar2 = fig.colorbar(dsd_plot, orientation="vertical", ax=ax2, cax=cbar_ax2)
    cbar2.set_ticklabels(cbar2.get_ticks(), fontsize=14)
    cbar2.set_label(label='Number Density \n Per Unit \n Volume \n ($m^{-3}$ $mm$)', fontsize=16)

    try:
        if outdir:
            plt.savefig(outdir + "xprecipradar.radclss.timeseries." + DATE + ".png", bbox_inches="tight", dpi=300)
        else:
            plt.savefig("xprecipradar.radclss.timeseries." + DATE + ".png", bbox_inches="tight", dpi=300)
        status = "SUCCESS: " + DATE + " timeseries plot"
    except:
        status = "FAILURE: " + DATE + " timeseries plot"
        
    # Free up memory and delete files read into memory
    del ld_accum, pluvio_accum, ds_resampled, radclss
    del fig, dbz_plot, dsd_plot

    return status


def radclss(volumes, serial=True, outdir=None, postprocess=True):
    """
    Extracted Radar Columns and In-Situ Sensors

    Utilizing Py-ART and ACT, extract radar columns above various sites and 
    collocate with in-situ ground based sensors.

    Within this verison of RadCLss, supported sensors are:
        - Pluvio Weighing Bucket Rain Gauge
        - Surface Meteorological Sensors (MET)
        - Laser Disdrometer (mutliple sites)
        - Radar Wind Profiler
        - Interpolated Radiosonde
        - Ceilometer

    Calls
    -----
    subset_points
    match_datasets_act

    Parameters
    ----------
    volumes : Dictionary
        Dictionary contianing files for each of the instruments, including
        all CMAC processed radar files per day. 
    
    Keywords
    --------
    serial : Boolean, Default = False
        Option to denote serial processing; used to start dask cluster for
        subsetting columns
    
    outdir : str, Default = None
        Option of specifying Location of where to write RadCLss files
    
    postprocess : Boolean, Default = True
        Option of creating timeseries plot for the processed RadCLss file. 

    Returns
    -------
    ds : Xarray Dataset
        Daily time-series of extracted columns saved into ARM formatted netCDF files. 
    """
    # Define variables to drop from RadCLss from the respective datastreams 
    discard_var = {'radar' : ['classification_mask',
                              'censor_mask',
                              'uncorrected_copol_correlation_coeff',
                              'uncorrected_differential_phase',
                              'uncorrected_differential_reflectivity',
                              'uncorrected_differential_reflectivity_lag_1',
                              'uncorrected_mean_doppler_velocity_h',
                              'uncorrected_mean_doppler_velocity_v',
                              'uncorrected_reflectivity_h',
                              'uncorrected_reflectivity_v',
                              'uncorrected_spectral_width_h',
                              'uncorrected_spectral_width_v',
                              'clutter_masked_velocity',
                              'gate_id',
                              'ground_clutter',
                              'partial_beam_blockage',
                              'cumulative_beam_blockage',
                              'unfolded_differential_phase',
                              'path_integrated_attenuation',
                              'path_integrated_differential_attenuation',
                              'unthresholded_power_copolar_v',
                              'unthresholded_power_copolar_h',
                              'specific_differential_phase',
                              'specific_differential_attenuation',
                              'reflectivity_v',
                              'reflectivity',
                              'mean_doppler_velocity_v',
                              'mean_doppler_velocity',
                              'differential_reflectivity_lag_1',
                              'differential_reflectivity',
                              'differential_phase'
                    ],
                    'met' : ['base_time', 
                             'time_offset', 
                             'time_bounds', 
                             'logger_volt',
                             'logger_temp', 
                             'qc_logger_temp', 
                             'lat', 
                             'lon', 
                             'alt', 
                             'qc_temp_mean',
                             'qc_rh_mean', 
                             'qc_vapor_pressure_mean', 
                             'qc_wspd_arith_mean', 
                             'qc_wspd_vec_mean',
                             'qc_wdir_vec_mean', 
                             'qc_pwd_mean_vis_1min', 
                             'qc_pwd_mean_vis_10min', 
                             'qc_pwd_pw_code_inst',
                             'qc_pwd_pw_code_15min', 
                             'qc_pwd_pw_code_1hr', 
                             'qc_pwd_precip_rate_mean_1min',
                             'qc_pwd_cumul_rain', 
                             'qc_pwd_cumul_snow', 
                             'qc_org_precip_rate_mean', 
                             'qc_tbrg_precip_total',
                             'qc_tbrg_precip_total_corr', 
                             'qc_logger_volt', 
                             'qc_logger_temp', 
                             'qc_atmos_pressure', 
                             'pwd_pw_code_inst', 
                             'pwd_pw_code_15min', 
                             'pwd_pw_code_1hr', 
                             'temp_std', 
                             'rh_std',
                             'vapor_pressure_std', 
                             'wdir_vec_std', 
                             'tbrg_precip_total', 
                             'org_precip_rate_mean',
                             'pwd_mean_vis_1min', 
                             'pwd_mean_vis_10min', 
                             'pwd_precip_rate_mean_1min', 
                             'pwd_cumul_rain',
                             'pwd_cumul_snow', 
                             'pwd_err_code'
                    ],
                    'sonde' : ['base_time', 
                               'time_offset', 
                               'lat', 
                               'lon', 
                               'qc_pres',
                               'qc_tdry', 
                               'qc_dp', 
                               'qc_wspd', 
                               'qc_deg', 
                               'qc_rh',
                               'qc_u_wind', 
                               'qc_v_wind', 
                               'qc_asc', 
                               "wstat", 
                               "asc"
                    ],
                    'pluvio' : ['base_time', 
                                'time_offset', 
                                'load_cell_temp', 
                                'heater_status',
                                'elec_unit_temp', 
                                'supply_volts', 
                                'orifice_temp', 
                                'volt_min',
                                'ptemp', 
                                'lat', 
                                'lon', 
                                'alt', 
                                'maintenance_flag', 
                                'reset_flag', 
                                'qc_rh_mean', 
                                'pluvio_status', 
                                'bucket_rt', 
                                'accum_total_nrt'
                    ],
                    'ldquants' : ['specific_differential_attenuation_xband20c',
                                  'specific_differential_attenuation_kaband20c',
                                  'specific_differential_attenuation_sband20c',
                                  'bringi_conv_stra_flag',
                                  'exppsd_slope',
                                  'norm_num_concen',
                                  'num_concen',
                                  'gammapsd_shape',
                                  'gammapsd_slope',
                                  'mean_doppler_vel_wband20c',
                                  'mean_doppler_vel_kaband20c',
                                  'mean_doppler_vel_xband20c',
                                  'mean_doppler_vel_sband20c',
                                  'specific_attenuation_kaband20c',
                                  'specific_attenuation_xband20c',
                                  'specific_attenuation_sband20c',
                                  'specific_differential_phase_kaband20c',
                                  'specific_differential_phase_xband20c',
                                  'specific_differential_phase_sband20c',
                                  'differential_reflectivity_kaband20c',
                                  'differential_reflectivity_xband20c',
                                  'differential_reflectivity_sband20c',
                                  'reflectivity_factor_wband20c',
                                  'reflectivity_factor_kaband20c',
                                  'reflectivity_factor_xband20c',
                                  'reflectivity_factor_sband20c',
                                  'time_offset',
                                  'base_time',
                                  'lat',
                                  'lon',
                                  'alt'
                    ]
    }
    print(volumes['date'] + " start subset-points: ", time.strftime("%H:%M:%S"))
    
    # Call Subset Points
    columns = []
    if serial == False:
        with LocalCluster(n_workers=4, processes=True, threads_per_worker=1, silence_logs=logging.ERROR,
                          ) as cluster, Client(cluster) as client:
            results = client.map(subset_points, volumes["radar"])
            for done_work in as_completed(results, with_results=False):
                try:
                    columns.append(done_work.result())
                except Exception as error:
                    log.exception(error)
    else:
        for rad in volumes['radar']:
            columns.append(subset_points(rad))
    # Assemble individual columns into single DataSet
    try:
        ds = xr.concat([data for data in columns if data], dim="time")
    except ValueError:
        ds = None
    # Free up Memory
    del columns

    # If successful column extraction, apply in-situ
    if ds:
        # Remove Global Attributes from the Column Extraction
        # Attributes make sense for single location, but not collection of sites.
        ds.attrs = {}
        # Remove the Base_Time variable from extracted column
        del ds['base_time']
        # Depending on how Dask is behaving, may be to resort time
        ds = ds.sortby("time")
        print(volumes['date'] + " finish subset-points: ", time.strftime("%H:%M:%S"))
    
        # Pluvio Weighing Bucket Rain Gauge
        if volumes['pluvio']:
            # Weighing Bucket Rain Gauge
            ds = match_datasets_act(ds, 
                                    volumes['pluvio'][0], 
                                    "M1", 
                                    discard=discard_var['pluvio'])

        if volumes['met_m1']:
            # Surface Meteorological Station
            ds = match_datasets_act(ds, 
                                    volumes['met_m1'][0], 
                                    "M1", 
                                    discard=discard_var['met'])
        
        if volumes['met_s20']:
            # Surface Meteorological Station
            ds = match_datasets_act(ds, 
                                    volumes['met_s20'][0], 
                                    "S20", 
                                    discard=discard_var['met'])

        if volumes['met_s30']:
            # Surface Meteorological Station
            ds = match_datasets_act(ds, 
                                    volumes['met_s30'][0], 
                                    "S30", 
                                    discard=discard_var['met'])

        if volumes['met_s40']:
            # Surface Meteorological Station
            ds = match_datasets_act(ds, 
                                    volumes['met_s40'][0], 
                                    "S40", 
                                    discard=discard_var['met'])

        if volumes['ld_m1']:
            # Laser Disdrometer - Main Site
            ds = match_datasets_act(ds, 
                                    volumes['ld_m1'][0], 
                                    "M1", 
                                    discard=discard_var['ldquants'],
                                    resample="mean")

        if volumes['ld_s30']:
            # Laser Disdrometer - Supplemental Site
            ds = match_datasets_act(ds, 
                                    volumes['ld_s30'][0], 
                                    "S30", 
                                    discard=discard_var['ldquants'],
                                    resample="mean")

        # ----------------
        # Check DOD - TBD
        # ----------------
        out_ds = ds.copy()
        
        # ------------
        # Save to File
        # ------------
        # write to file
        try:
            if outdir:
                out_ds.to_netcdf(outdir + 'bnfcsapr2radclssS3.c2.' + volumes['date'] + '.000000.nc')
            else:
                out_ds.to_netcdf('bnfcsapr2radclssS3.c2.' + volumes['date'] + '.000000.nc')
            status = ": RadCLss SUCCESS: " + volumes['date']
        except ValueError:
            status = ": RadCLss FAILURE: " + volumes['date']

        # create timeseries plot
        if postprocess == True:
            try:
                plot_status = create_radclss_figure(out_ds, outdir=outdir)
                print(plot_status)
            except ValueError:
                print("PLOT FAILURE: " + volumes['date'])
    
        # free up memory
        del ds, out_ds

    else:
        # There is no column extraction
        status = ": RadCLss FAILURE (All Columns Failed to Extract): "
        del ds

    return status

def main(args):
    print("process start time: ", time.strftime("%H:%M:%S"))
    # Define the directories where the CSAPR2 and In-Situ files are located.
    RADAR_DIR = args.radar_dir + '%s/' % args.month
    INSITU_DIR = args.insitu_dir + '%s/' %args.month
    OUT_PATH = args.outdir + '/%s/' % args.month
    print("OUTPATH: ", OUT_PATH)

    # Define an output directory for downloaded ground instrumentation
    insitu_stream = {'bnfmetM1.b1' : INSITU_DIR + 'bnfmetM1.b1',
                    'bnfmetS20.b1' : INSITU_DIR + "bnfmetS20.b1",
                    "bnfmetS30.b1" : INSITU_DIR + "bnfmetS30.b1",
                    "bnfmetS40.b1" : INSITU_DIR + "bnfmetS40.b1",
                    "bnfsondewnpnM1.b1" : INSITU_DIR + "bnfsondewnpnM1.b1",
                    "bnfwbpluvio2M1.a1" : INSITU_DIR + "bnfwbpluvio2M1.a1",
                    "bnfldquantsM1.c1" : INSITU_DIR + "bnfldquantsM1.c1",
                    "bnfldquantsS30.c1" : INSITU_DIR + "bnfldquantsS30.c1",
                    }

    # define the number of days within the month
    d0 = datetime.datetime(year=int(args.month[0:4]), month=int(args.month[4:7]), day=1)
    d1 = datetime.datetime(year=int(args.month[0:4]), month=int(args.month[4:7])+1, day=1)
    volumes = {'date': [], 
               'radar' : [], 
               'pluvio' : [], 
               'met_m1' : [],
               'met_s20' : [],
               'met_s30' : [],
               'met_s40' : [],
               'ld_m1' : [], 
               'ld_s30' : [], 
               'sonde' : [], 
    }
    
    # Subset dictionary for desired indice 
    def ith_val_subdict(input_dict, i):
        return {k: v[i] for k, v in input_dict.items()}
    
    # iterate through files and collect together
    if args.array is True:
        day_of_month = args.month + args.day
        print("day of month: ", day_of_month)
        volumes['date'].append(day_of_month)
        volumes['radar'].append(sorted(glob.glob(RADAR_DIR + day_of_month + '*')))
        volumes['pluvio'].append(sorted(glob.glob(insitu_stream['bnfwbpluvio2M1.a1'] + day_of_month + '*.nc')))
        volumes['met_m1'].append(sorted(glob.glob(insitu_stream['bnfmetM1.b1'] + day_of_month + '*.cdf')))
        volumes['met_s20'].append(sorted(glob.glob(insitu_stream['bnfmetS20.b1'] + day_of_month + '*.cdf')))
        volumes['met_s30'].append(sorted(glob.glob(insitu_stream['bnfmetS30.b1'] + day_of_month + '*.cdf')))
        volumes['met_s40'].append(sorted(glob.glob(insitu_stream['bnfmetS40.b1'] + day_of_month + '*.cdf')))
        volumes['ld_m1'].append(sorted(glob.glob(insitu_stream['bnfldquantsM1.c1'] + day_of_month + '*.cdf')))
        volumes['ld_s30'].append(sorted(glob.glob(insitu_stream['bnfldquantsS30.b1'] + day_of_month + '*.cdf')))
        volumes['sonde'].append(sorted(glob.glob(insitu_stream['bnfsondewnpnM1.b1'] + day_of_month + '*.cdf')))
    else:
        for i in range((d1-d0).days):
            if i < 9:
                day_of_month = args.month + '0' + str(i+1)
                volumes['date'].append(day_of_month)
                volumes['radar'].append(sorted(glob.glob(RADAR_DIR + day_of_month + '*')))
                volumes['pluvio'].append(sorted(glob.glob(insitu_stream['bnfwbpluvio2M1.a1'] + day_of_month + '*.nc')))
                volumes['met_m1'].append(sorted(glob.glob(insitu_stream['bnfmetM1.b1'] + day_of_month + '*.cdf')))
                volumes['met_s20'].append(sorted(glob.glob(insitu_stream['bnfmetS20.b1'] + day_of_month + '*.cdf')))
                volumes['met_s30'].append(sorted(glob.glob(insitu_stream['bnfmetS30.b1'] + day_of_month + '*.cdf')))
                volumes['met_s40'].append(sorted(glob.glob(insitu_stream['bnfmetS40.b1'] + day_of_month + '*.cdf')))
                volumes['ld_m1'].append(sorted(glob.glob(insitu_stream['bnfldquantsM1.c1'] + day_of_month + '*.cdf')))
                volumes['ld_s30'].append(sorted(glob.glob(insitu_stream['bnfldquantsS30.b1'] + day_of_month + '*.cdf')))
                volumes['sonde'].append(sorted(glob.glob(insitu_stream['bnfsondewnpnM1.b1'] + day_of_month + '*.cdf')))
            else:
                day_of_month = args.month + str(i+1)
                volumes['date'].append(day_of_month)
                volumes['radar'].append(sorted(glob.glob(RADAR_DIR + day_of_month + '*')))
                volumes['pluvio'].append(sorted(glob.glob(insitu_stream['bnfwbpluvio2M1.a1'] + day_of_month + '*.nc')))
                volumes['met_m1'].append(sorted(glob.glob(insitu_stream['bnfmetM1.b1'] + day_of_month + '*.cdf')))
                volumes['met_s20'].append(sorted(glob.glob(insitu_stream['bnfmetS20.b1'] + day_of_month + '*.cdf')))
                volumes['met_s30'].append(sorted(glob.glob(insitu_stream['bnfmetS30.b1'] + day_of_month + '*.cdf')))
                volumes['met_s40'].append(sorted(glob.glob(insitu_stream['bnfmetS40.b1'] + day_of_month + '*.cdf')))
                volumes['ld_m1'].append(sorted(glob.glob(insitu_stream['bnfldquantsM1.c1'] + day_of_month + '*.cdf')))
                volumes['ld_s30'].append(sorted(glob.glob(insitu_stream['bnfldquantsS30.b1'] + day_of_month + '*.cdf')))
                volumes['sonde'].append(sorted(glob.glob(insitu_stream['bnfsondewnpnM1.b1'] + day_of_month + '*.cdf')))
 
    # Send volume to RadClss for processing
    for i in range(len(volumes['date'])):
        nvol = ith_val_subdict(volumes, i)
        if nvol["radar"]:
            if args.verbose:
                print("serial - ", args.serial)
                print(volumes['date'][i], nvol["radar"])
            status = radclss(nvol, outdir=OUT_PATH, serial=args.serial)
            print(status)
 
    print("processing finished: ", time.strftime("%H:%M:%S"))
    # free up memory
    del volumes, nvol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Matched Radar Columns and In-Situ Sensors (RadCLss) Processing." +
            "Extracts Radar columns above a given site and collocates with in-situ sensors")

    parser.add_argument("--month",
                        default="202203",
                        dest='month',
                        type=str,
                        help="[str|YYYMM format] Specific Month to Process"
    )

    parser.add_argument("--day",
                        default="01",
                        dest='day',
                        type=str,
                        help="[str|DD format] Specific Day to Process. Checks for `array` first"
    )

    parser.add_argument("--array",
                        default=False,
                        dest="array",
                        type=bool,
                        help="[bool|default=False] If Set, check for specific days to process"
    )
    
    parser.add_argument("--day",
                        default="01",
                        dest='day',
                        type=str,
                        help="[str|DD format] Specific Day to Process. Checks for `array` first"
    )
    
    parser.add_argument("--serial",
                        default=True,
                        dest='serial',
                        type=bool,
                        help="[bool|default=False] Process in Serial for testing"
    )
    
    parser.add_argument("--outdir",
                        default='/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradclssS2.c2',
                        dest='outdir',
                        type=str,
                        help="[str] Specific directory to write RadCLss to"
    )
    
    parser.add_argument("--radar_dir",
                        default='/nfs/gce/globalscratch/obrienj/bnf-cmac-r4/',
                        dest='radar_dir',
                        type=str,
                        help="[str] Specific directory where the CMAC files are located"
    )

    parser.add_argument("--insitu_dir",
                        default='/nfs/gce/globalscratch/obrienj/bnf-cmac-r4/in_situ',
                        dest='radar_dir',
                        type=str,
                        help="[str] Specific directory where the in-situ files are located"
    )
    
    parser.add_argument("--postprocessing",
                        default=True,
                        dest="postproc",
                        type=bool,
                        help="[bool|default=True] Create timeseries figures using generated RadClss files"
    )
    
    parser.add_argument("--verbose",
                        default=False,
                        dest="verbose",
                        type=bool,
                        help="[bool|default=False] Display file paths"
    )
    
    args = parser.parse_args()

    main(args)

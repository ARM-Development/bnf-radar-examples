import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import glob
import sys
import time
import datetime
import argparse
import logging
import dask

import numpy as np
import xarray as xr
import pandas as pd

from dask.distributed import Client, LocalCluster, wait, as_completed, fire_and_forget
dask.config.set({'logging.distributed': 'error'})

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
    S13 = [34.343889, -87.350556]

    sites    = ["M1", "S4", "S20", "S30", "S40", "S13"]
    site_alt = [293, 197, 178, 183, 236, 286]

    # Zip these together!
    lats, lons = list(zip(M1,
                          S4,
                          S20,
                          S30,
                          S40,
                          S13))
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
                        # check for valid heights
                        valid = np.isfinite(da["height"])
                        n_valid = int(valid.sum())
                        if n_valid > 0:
                            da = da.sel(height=valid).sortby("height").interp(height=np.arange(500, 8500, 250))
                        else:
                            target_height = xr.DataArray(np.arange(500, 8500, 250), dims="height", name="height")
                            da = da.reindex(height=target_height)

                    # Add the latitude and longitude of the extracted column
                    da["lat"], da["lon"] = lat, lon
                    # Convert timeoffsets to timedelta object and precision on datetime64
                    da.time_offset.data = da.time_offset.values.astype("timedelta64[s]")
                    da.base_time.data = da.base_time.values.astype("datetime64[s]")
                    # Time is based off the start of the radar volume
                    da["gate_time"] = da.base_time.values + da.isel(height=0).time_offset.values
                    column_list.append(da)
        
                # Concatenate the extracted radar columns for this scan across all sites    
                ds = xr.concat([data for data in column_list if data], dim='station')
                ds["station"] = sites
                # Assign the Main and Supplemental Site altitudes
                ds = ds.assign(alt=("station", site_alt))
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
                ds.lat.attrs.update(long_name='Latitude of BNF AMF-3 Ground Observation Site',
                                         units='Degrees North')
                ds.lon.attrs.update(long_name='Longitude of BNF AMF-3 Ground Observation Site',
                                          units='Degrees East')
                ds.alt.attrs.update(long_name="Altitude above mean sea level for each station",
                                          units="m")
                # delete the radar to free up memory
                del radar, column_list, da
            else:
                # delete the rhi file
                del radar
        else:
            del radar
    return ds

def match_datasets_act(column, 
                       ground, 
                       site, 
                       discard, 
                       resample='sum', 
                       DataSet=False,
                       prefix=None):
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

    prefix : str
        prefix for the desired spelling of variable names for the input
        datastream (to fix duplicate variable names between instruments)
             
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
        # check if a list containing new variable names exists. 
        if prefix:
            grd_ds = grd_ds.rename_vars({v: f"{prefix}{v}" for v in grd_ds.data_vars})
        
    # Remove Base_Time before Resampling Data since you can't force 1 datapoint to 5 min sum
    if 'base_time' in grd_ds.data_vars:
        del grd_ds['base_time']
        
    # Check to see if height is a dimension within the ground instrumentation. 
    # If so, first interpolate heights to match radar, before interpolating time.
    if 'height' in grd_ds.dims:
        grd_ds = grd_ds.interp(height=np.arange(3150, 10050, 50), method='linear')
        
    # Resample the ground data to 5 min and interpolate to the CSU X-Band time. 
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
    
    # Add SAIL site location as a dimension for the Pluvio data
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

def adjust_radclss_dod(radclss, dod):
    """
    Adjust the RadCLss DataSet to include missing datastreams

    Parameters
    ----------
    radclss : Xarray DataSet
        extracted columns and in-situ sensor file
    dod : Xarray DataSet
        expected datastreams and data standards for RadCLss

    returns
    -------
    radclss : Xarray DataSet
        Corrected RadCLss that has all expected parmeters and attributes
    """
    # Supplied DOD has correct data attributes and all required parameters. 
    # Update the RadCLss dataset variable values with the DOD dataset. 
    print("\n Variables not within RADClss ds:")
    for var in dod.variables:
        # Make sure the variable isn't a dimension
        if var not in dod.dims:
            # check to see if variable is within RadCLss
            # note: it may not be if file is missing.
            if var not in radclss.variables:
                print(var)
                new_size = []
                for dim in dod[var].dims:
                    if dim == "time":
                        new_size.append(radclss.sizes['time'])
                    else:
                        new_size.append(dod.sizes[dim])
                    #new_data = np.full(())
                # create a new array to hold the updated values
                new_data = np.full(new_size, dod[var].data[0])
                # create a new DataArray and add back into RadCLss
                new_array = xr.DataArray(new_data, dims=dod[var].dims)
                new_array.attrs = dod[var].attrs
                radclss[var] = new_array
                
                # clean up some saved values
                del new_size, new_data, new_array

    # Adjust the radclss time attributes
    if hasattr(radclss['time'], "units"):
        del radclss["time"].attrs["units"]
    if hasattr(radclss['time_offset'], "units"):
        del radclss["time_offset"].attrs["units"]
    if hasattr(radclss['base_time'], "units"):
        del radclss["base_time"].attrs["units"]

    # reorder the DataArrays to match the ARM Data Object Identifier 
    first = ["base_time", "time_offset", "time", "height", "station", "gate_time"]           # the two you want first
    last  = ["lat", "lon", "alt"]   # the three you want last

    # Keep only data variables, preserve order, and drop the ones already in first/last
    middle = [v for v in radclss.data_vars if v not in first + last]

    ordered = first + middle + last
    radclss = radclss[ordered]

    # Update the global attributes
    radclss.drop_attrs()
    radclss.attrs = dod.attrs
    radclss.attrs['vap_name'] = ""
    radclss.attrs['command_line'] = "python bnf_radclss.py --serial True --array True"
    radclss.attrs['dod_version'] = "csapr2radclss-c2-1.1"
    radclss.attrs['site_id'] = "bnf"
    radclss.attrs['platform_id'] = "csapr2radclss"
    radclss.attrs['facility_id'] = "S3"
    radclss.attrs['data_level'] = "c2"
    radclss.attrs['location_description'] = "Southeast U.S. in Bankhead National Forest (BNF), Decatur, Alabama"
    radclss.attrs['datastream'] = "bnfcsapr2radclssS3.c2"
    radclss.attrs['input_datastreams'] = "bnfcsapr2cmacS3.c1"
    radclss.attrs['history'] = ("created by user jrobrien on machine cumulus.ccs.ornl.gov at " + 
                                 str(datetime.datetime.now()) +
                                " using Py-ART and ACT"
    )
    print("\n")
    # return radclss, close DOD file
    del dod

    return radclss

def radclss(volumes, serial=True, outdir=None, dod_file=None):
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

    dod_file : str, Default = None
        Option to supply a Data Object Description file to verify standards

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
                          'differential_phase',
                          'normalized_coherent_power',
                          'normalized_coherent_power_v',
                          'signal_to_noise_ratio_copolar_h',
                          'signal_to_noise_ratio_copolar_v',
                          'specific_attenuation',
                          'spectral_width',
                          'spectral_width_v',
                          'sounding_temperature',
                          'signal_to_noise_ratio',
                          'velocity_texture',
                          'simulated_velocity',
                          'height_over_iso0'
                    ],
                    'met' : ['base_time', 'time_offset', 'time_bounds', 'logger_volt',
                        'logger_temp', 'qc_logger_temp', 'lat', 'lon', 'alt', 'qc_temp_mean',
                        'qc_rh_mean', 'qc_vapor_pressure_mean', 'qc_wspd_arith_mean', 'qc_wspd_vec_mean',
                        'qc_wdir_vec_mean', 'qc_pwd_mean_vis_1min', 'qc_pwd_mean_vis_10min', 'qc_pwd_pw_code_inst',
                        'qc_pwd_pw_code_15min', 'qc_pwd_pw_code_1hr', 'qc_pwd_precip_rate_mean_1min',
                        'qc_pwd_cumul_rain', 'qc_pwd_cumul_snow', 'qc_org_precip_rate_mean', 'qc_tbrg_precip_total',
                        'qc_tbrg_precip_total_corr', 'qc_logger_volt', 'qc_logger_temp', 'qc_atmos_pressure', 
                        'pwd_pw_code_inst', 'pwd_pw_code_15min', 'pwd_pw_code_1hr', 'temp_std', 'rh_std',
                        'vapor_pressure_std', 'wdir_vec_std', 'tbrg_precip_total', 'org_precip_rate_mean',
                        'pwd_mean_vis_1min', 'pwd_mean_vis_10min', 'pwd_precip_rate_mean_1min', 'pwd_cumul_rain',
                        'pwd_cumul_snow', 'pwd_err_code'
                    ],
                    'sonde' : ['base_time', 'time_offset', 'lat', 'lon', 'qc_pres',
                           'qc_tdry', 'qc_dp', 'qc_wspd', 'qc_deg', 'qc_rh',
                           'qc_u_wind', 'qc_v_wind', 'qc_asc', "wstat", "asc"
                    ],
                    'pluvio' : ['base_time', 'time_offset', 'load_cell_temp', 'heater_status',
                            'elec_unit_temp', 'supply_volts', 'orifice_temp', 'volt_min',
                            'ptemp', 'lat', 'lon', 'alt', 'maintenance_flag', 'reset_flag', 
                            'qc_rh_mean', 'pluvio_status', 'bucket_rt', 'accum_total_nrt'
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
                              'bringi_conv_stra_flag',
                              'time_offset',
                              'base_time',
                              'lat',
                              'lon',
                              'alt'
                    ],
                    'vdisquants' : ['specific_differential_attenuation_xband20c',
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
                                'bringi_conv_stra_flag',
                                'time_offset',
                                'base_time',
                                'lat',
                                'lon',
                                'alt'
                    ],
                    'wxt' : ['base_time',
                         'time_offset',
                         'time_bounds',
                         'qc_temp_mean',
                         'temp_std',
                         'rh_mean',
                         'qc_rh_mean',
                         'rh_std',
                         'atmos_pressure',
                         'qc_atmos_pressure',
                         'wspd_arith_mean',
                         'qc_wspd_arith_mean',
                         'wspd_vec_mean',
                         'qc_wspd_vec_mean',
                         'wdir_vec_mean',
                         'qc_wdir_vec_mean',
                         'wdir_vec_std',
                         'qc_wxt_precip_rate_mean',
                         'qc_wxt_cumul_precip',
                         'logger_volt',
                         'qc_logger_volt',
                         'logger_temp',
                         'qc_logger_temp',
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
        # Concatenate all extracted columns across time dimension to form daily timeseries
        ds = xr.concat([data for data in columns if data], dim="time")
        ds['time'] = ds.sel(station="M1").base_time
        ds['time_offset'] = ds.sel(station="M1").base_time
        ds['base_time'] = ds.sel(station="M1").isel(time=0).base_time
        ds['lat'] = ds.isel(time=0).lat
        ds['lon'] = ds.isel(time=0).lon
        ds['alt'] = ds.isel(time=0).alt
        # Remove all the unused CMAC variables
        ds = ds.drop_vars(discard_var["radar"])
        # Drop duplicate latitude and longitude
        ds = ds.drop_vars(['latitude', 'longitude'])
    except ValueError:
        ds = None
        if client:
            client.close()
    # Free up Memory
    del columns

    # If successful column extraction, apply in-situ
    if ds:
        # Depending on how Dask is behaving, may be to resort time
        ds = ds.sortby("time")
        print(volumes['date'] + " finish subset-points: ", time.strftime("%H:%M:%S"))
    
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
            
        if volumes['sonde']:
            print(volumes['sonde'][0])
            # Read in the file using ACT
            grd_ds = act.io.read_arm_netcdf(volumes['sonde'], 
                                            cleanup_qc=True,
                                            drop_variables=discard_var['sonde'])
            # Default are Lazy Arrays; convert for matching with column
            grd_ds = grd_ds.compute()
            # check if a list containing new variable names exists.
            prefix = "sonde_"
            grd_ds = grd_ds.rename_vars({v: f"{prefix}{v}" for v in grd_ds.data_vars})
            # Match to the columns
            ds = match_datasets_act(ds, 
                                    grd_ds, 
                                    "M1",
                                    discard=discard_var['sonde'],
                                    DataSet=True,
                                    resample="mean")
            # clean up
            del grd_ds
        
        # Pluvio Weighing Bucket Rain Gauge
        if volumes['pluvio']:
            # Weighing Bucket Rain Gauge
            ds = match_datasets_act(ds, 
                                    volumes['pluvio'][0], 
                                    "M1", 
                                    discard=discard_var['pluvio'])

        if volumes['ld_m1']:
            # Laser Disdrometer - Main Site
            ds = match_datasets_act(ds, 
                                    volumes['ld_m1'][0], 
                                    "M1", 
                                    discard=discard_var['ldquants'],
                                    resample="mean",
                                    prefix="ldquants_")

        if volumes['ld_s30']:
            # Laser Disdrometer - Supplemental Site
            ds = match_datasets_act(ds, 
                                    volumes['ld_s30'][0], 
                                    "S30", 
                                    discard=discard_var['ldquants'],
                                    resample="mean",
                                    prefix="ldquants_")
            
        if volumes['vd_m1']:
            # Laser Disdrometer - Supplemental Site
            ds = match_datasets_act(ds, 
                                    volumes['vd_m1'][0], 
                                    "M1", 
                                    discard=discard_var['vdisquants'],
                                    resample="mean",
                                    prefix="vdisquants_")
        
        if volumes['wxt_s13']:
            # Laser Disdrometer - Supplemental Site
            ds = match_datasets_act(ds, 
                                    volumes['wxt_s13'][0], 
                                    "S13", 
                                    discard=discard_var['wxt'],
                                    resample="mean")

        # ---------------------------------------------------------------
        # Cumulus cannot access the DOD API, Requires locally stored file
        # ---------------------------------------------------------------
        # verify the correct dimension order
        ds = ds.transpose("time", "height", "station")
        if dod_file:
            try:
                dod = xr.open_dataset(dod_file)
                # verify the correct dimension order
                ds = adjust_radclss_dod(ds, dod)
            except ValueError as e:
                print(f"Error: {e}")
                print(f"Error type: {type(e).__name__}")
                print("WARNING: Unable to Verify DOD")
        
        # ------------
        # Save to File
        # ------------
        # Define RADCLss keys to skip
        keys_to_skip = ['alt', 
                        'lat', 
                        'lon', 
                        'gate_time', 
                        'station', 
                        'base_time',
                        'time_offset',
                        'time']

        # Define the keys that have int MVC
        diff_keys = ['vdisquants_rain_rate',
                     'vdisquants_reflectivity_factor_cband20c',
                     'vdisquants_differential_reflectivity_cband20c',
                     'vdisquants_specific_differential_phase_cband20c',
                     'vdisquants_specific_attenuation_cband20c',
                     'vdisquants_med_diameter',
                     'vdisquants_mass_weighted_mean_diameter',
                     'vdisquants_lwc',
                     'vdisquants_total_droplet_concentration',
                     'vdisquants_mean_doppler_vel_cband20c',
                     'vdisquants_specific_differential_attenuation_cband20c',
                     'wxt_temp_mean',
                     'wxt_precip_rate_mean',
                     'wxt_cumul_precip']

        # Drop the missing value code attribute from the diff_keys
        for key in diff_keys:
            if hasattr(ds[key], "missing_value"):
                del ds[key].attrs['missing_value']

        # Create a dictionary to hold encoding information
        filtered_keys = [key for key in ds.keys() if key not in {*keys_to_skip, *diff_keys}]
        encoding_dict = {key : {"_FillValue" : -9999.} for key in filtered_keys}
        diff_dict = {key : {"_FillValue" : -9999} for key in diff_keys}
        encoding_dict.update(diff_dict)

        try:
            if outdir:
                out_path = outdir + 'bnfcsapr2radclssS3.c2.' + volumes['date'] + '.000000.nc'
                # Convert DataSet to ACT DataSet for writing
                ds_out = act.io.WriteDataset(ds)
                # If FillValue set to True, will just apply NaNs instead of encodings
                ds_out.write_netcdf(path=out_path,
                                    FillValue=False,
                                    encoding=encoding_dict)
                # Free up Memory
                del ds_out, encoding_dict,diff_dict, keys_to_skip, diff_keys
                status = ": RadCLss SUCCESS: " + volumes['date']
            else:
                out_path = 'bnfcsapr2radclssS3.c2.' + volumes['date'] + '.000000.nc'
                # Convert DataSet to ACT DataSet for writing
                ds_out = act.io.WriteDataset(ds)
                # If FillValue set to True, will just apply NaNs instead of encodings
                ds_out.write_netcdf(path=out_path,
                                    FillValue=False,
                                    encoding=encoding_dict)
                # Free up Memory
                del ds_out, encoding_dict,diff_dict, keys_to_skip, diff_keys
                status = ": RadCLss SUCCESS: " + volumes['date']
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")
            status = ": RadCLss Write FAILURE: " + volumes['date']
  
        # free up memory
        del ds

    else:
        # There is no column extraction
        status = ": RadCLss FAILURE (All Columns Failed to Extract): "
        del ds

    return status

def main(args):
    if args.verbose:
        print(args)
    print("process start time: ", time.strftime("%H:%M:%S"))
    # Define the directories where the CSAPR2 and In-Situ files are located.
    RADAR_DIR = args.radar_dir 
    INSITU_DIR = args.insitu_dir
    OUT_PATH = args.outdir
    if args.verbose:
        print("\n")
        print("RADAR_PATH", RADAR_DIR)
        print("INSITU_PATH", INSITU_DIR)
        print("OUTPATH: ", OUT_PATH)

    # Define an output directory for downloaded ground instrumentation
    insitu_stream = {'bnfmetM1.b1' : INSITU_DIR + "bnfmetM1.b1/*",
                    'bnfmetS20.b1' : INSITU_DIR + "bnfmetS20.b1/*",
                    "bnfmetS30.b1" : INSITU_DIR + "bnfmetS30.b1/*",
                    "bnfmetS40.b1" : INSITU_DIR + "bnfmetS40.b1/*",
                    "bnfsondewnpnM1.b1" : INSITU_DIR + "bnfsondewnpnM1.b1/*",
                    "bnfwbpluvio2M1.a1" : INSITU_DIR + "bnfwbpluvio2M1.a1/*",
                    "bnfldquantsM1.c1" : INSITU_DIR + "bnfldquantsM1.c1/*",
                    "bnfldquantsS30.c1" : INSITU_DIR + "bnfldquantsS30.c1/*",
                    "bnfvdisquantsM1.c1" : INSITU_DIR + "bnfvdisquantsM1.c1/*",
                    "bnfmetwxtS13.b1" : INSITU_DIR + "bnfmetwxtS13.b1/*"
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
               'vd_m1' : [],
               'sonde' : [],
               'wxt_s13' : [], 
    }
    
    # Subset dictionary for desired indice 
    def ith_val_subdict(input_dict, i):
        return {k: v[i] for k, v in input_dict.items()}
    
    # iterate through files and collect together
    if args.array is True:
        day_of_month = args.month + args.day
        if args.verbose:
            print("day of month: ", day_of_month)
        volumes['date'].append(day_of_month)
        volumes['radar'].append(sorted(glob.glob(RADAR_DIR + '*' + day_of_month + '*')))
        volumes['pluvio'].append(sorted(glob.glob(insitu_stream['bnfwbpluvio2M1.a1'] + day_of_month + '*.nc')))
        volumes['met_m1'].append(sorted(glob.glob(insitu_stream['bnfmetM1.b1'] + day_of_month + '*.cdf')))
        volumes['met_s20'].append(sorted(glob.glob(insitu_stream['bnfmetS20.b1'] + day_of_month + '*.cdf')))
        volumes['met_s30'].append(sorted(glob.glob(insitu_stream['bnfmetS30.b1'] + day_of_month + '*.cdf')))
        volumes['met_s40'].append(sorted(glob.glob(insitu_stream['bnfmetS40.b1'] + day_of_month + '*.cdf')))
        volumes['ld_m1'].append(sorted(glob.glob(insitu_stream['bnfldquantsM1.c1'] + day_of_month + '*.nc')))
        volumes['ld_s30'].append(sorted(glob.glob(insitu_stream['bnfldquantsS30.c1'] + day_of_month + '*.nc')))
        volumes['vd_m1'].append(sorted(glob.glob(insitu_stream["bnfvdisquantsM1.c1"] + day_of_month + '*.nc')))
        volumes['sonde'].append(sorted(glob.glob(insitu_stream['bnfsondewnpnM1.b1'] + day_of_month + '*.cdf')))
        volumes['wxt_s13'].append(sorted(glob.glob(insitu_stream['bnfmetwxtS13.b1'] + day_of_month + '*.nc')))
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
                volumes['ld_s30'].append(sorted(glob.glob(insitu_stream['bnfldquantsS30.c1'] + day_of_month + '*.cdf')))
                volumes['vd_m1'].append(sorted(glob.glob(insitu_stream["bnfvdisquantsM1.c1"] + day_of_month + '*.nc')))
                volumes['sonde'].append(sorted(glob.glob(insitu_stream['bnfsondewnpnM1.b1'] + day_of_month + '*.cdf')))
                volumes['wxt_s13'].append(sorted(glob.glob(insitu_stream['bnfmetwxtS13.b1'] + day_of_month + '*.nc')))
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
                volumes['ld_s30'].append(sorted(glob.glob(insitu_stream['bnfldquantsS30.c1'] + day_of_month + '*.cdf')))
                volumes['vd_m1'].append(sorted(glob.glob(insitu_stream["bnfvdisquantsM1.c1"] + day_of_month + '*.nc')))
                volumes['sonde'].append(sorted(glob.glob(insitu_stream['bnfsondewnpnM1.b1'] + day_of_month + '*.cdf')))
                volumes['wxt_s13'].append(sorted(glob.glob(insitu_stream['bnfmetwxtS13.b1'] + day_of_month + '*.nc')))
 
    if args.verbose:
        print("\n")
        print("in-situ files located: ")
        for field in volumes:
            if field != "radar":
                print(volumes[field])
        print("\n")

    # Send volume to RadClss for processing
    for i in range(len(volumes['date'])):
        nvol = ith_val_subdict(volumes, i)
        if nvol["radar"]:
            status = radclss(nvol, 
                             outdir=OUT_PATH, 
                             serial=args.serial,
                             dod_file=args.dod_file)
            print(status)
 
    print("processing finished: ", time.strftime("%H:%M:%S"))
    # free up memory
    del volumes, nvol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Matched Radar Columns and In-Situ Sensors (RadCLss) Processing." +
            "Extracts Radar columns above a given site and collocates with in-situ sensors")

    parser.add_argument("--month",
                        default="202503",
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
   
    parser.add_argument("--serial",
                        default=False,
                        dest="serial",
                        type=bool,
                        help="[bool|default=False] Process in Serial for testing"
    )
    
    parser.add_argument("--outdir",
                        default='/gpfs/wolf2/arm/atm124/proj-shared/bnf/bnfcsapr2radclssS3.c2/',
                        dest='outdir',
                        type=str,
                        help="[str] Specific directory to write RadCLss to"
    )
    
    parser.add_argument("--radar_dir",
                        default='/gpfs/wolf2/arm/atm124/proj-shared/bnf/bnfcsapr2cmacS3.c1/',
                        dest='radar_dir',
                        type=str,
                        help="[str] Specific directory where the CMAC files are located"
    )

    parser.add_argument("--insitu_dir",
                        default='/gpfs/wolf2/arm/atm124/proj-shared/bnf/in_situ/',
                        dest='insitu_dir',
                        type=str,
                        help="[str] Specific directory where the in-situ files are located"
    )
       
    parser.add_argument("--verbose",
                        default=False,
                        dest="verbose",
                        type=bool,
                        help="[bool|default=False] Display file paths"
    )
    
    parser.add_argument("--dod_file",
                        default="/ccsopen/home/jrobrien/git-repos/bnf-radar-examples/notebooks/data/radclss/bnf-csapr2-radclss.dod.v1.1.nc",
                        dest="dod_file",
                        type=str,
                        help="[str] RadCLss DOD file"
    )
    
    args = parser.parse_args()

    main(args)

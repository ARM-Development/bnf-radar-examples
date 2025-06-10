import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import glob
import sys
import time
import datetime
import argparse
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

#-----------------
# Define Functions
#-----------------

def create_radclss_timeseries(radclss,
                              field="corrected_reflectivity", 
                              vmin=-5, 
                              vmax=65, 
                              dis_site="M1",
                              rheight=750, 
                              outdir="./"):
    """
    With the RadCLss product, generate a timeseries of radar reflectivity factor, 
    particle size distribution and cumuluative precipitaiton 
    for the ARM SAIL M1 Site. 

    This timeseries quick is to serve as a means for evaluating the RadCLss product.

    Input
    -----
    radclss : str
        Filepath to the RadCLss file.
    field : str
        Specific CMAC field to display extracted column of
    vmin : int
        Minimum value to display between all subplots for the specific radar
        parameter
    vmax : int
        Maximum value to display between all subplots for the specific radar
        parameter
    dis_site : str
        Identifer of the supported location for lat/lon slices
    height : int
        Column height to compare against in-situ sensors for precipitation accumulation. 
    outdir : str
        Path to desired output directory. If not supplied, assumes current working directory.

    Output
    ------
    timeseries : png
        Saved image of the RadCLss timeseris
    """
    # Create the figure
    fig = plt.figure(figsize=(16, 8))
    print(radclss)
    # read the RadCLss file
    try:
        ds = xr.open_dataset(radclss[0], decode_timedelta=False)
    except Exception as e:
        # 'e' will contain the error object
        print(f"An error occurred: {e}")
        print(f"Error type: {type(e)}")
        return
    
    #-----------------------------------------------
    # Side Plot A - Display the RadClss Radar Field
    #-----------------------------------------------
    # Top right hand subplot - Radar TimeSeries
    ax2 = fig.add_subplot(131)

    ds[field].sel(station=dis_site).plot(x="time", 
                                         ax=ax2, 
                                         cmap="ChaseSpectral",
                                         vmin=vmin,
                                         vmax=vmax
    )
    
    ax2.set_title("BNF Site: " + dis_site)
    ax2.set_ylabel("Height [m]")
    ax2.set_xlabel("Time [UTC]")

    #--------------------------------------
    # Side Plot B - Display the Rain Rates
    #--------------------------------------
    # Top right hand subplot - Radar TimeSeries
    ax3 = fig.add_subplot(132)

    # CMAC derived rain rate
    ds["rain_rate_A"].sel(station=dis_site).sel(
        height=rheight, method="nearest").plot(x="time", 
                                               ax=ax3,
                                               label="CMAC"
        )

    # Pluvio2 Weighing Bucket Rain Gauge
    if dis_site == "M1":
        ds["intensity_rtnrt"].sel(station=dis_site).plot(x="time",
                                                         ax=ax3,
                                                         label="PLUVIO2"
        )

    # LDQUANTS derived rain rate
    if dis_site == "M1" or dis_site == "S30":
        ds["rain_rate"].sel(station=dis_site).plot(x="time",
                                                   ax=ax3,
                                                   label="LDQUANTS"
        )
            
    ax3.set_title(" ")
    ax3.set_ylabel("Precipitation Rate [mm/hr]")
    ax3.set_xlabel("Time [UTC]")
    ax3.legend(loc="upper right")
    ax3.grid(True)
    # Add a blank space next to the subplot to shape it as the above plot
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="3%", pad=1.05)
    cax.set_visible(False)

    #------------------------------------------
    # Side Plot C - Precipitation Accumulation
    #------------------------------------------
    ax4 = fig.add_subplot(133)
   
    # CMAC Accumulated Rain Rates
    radar_accum = act.utils.accumulate_precip(ds.sel(station=dis_site).isel(height=rheight), 'rain_rate_A').compute()
    # CMAC Accumulated Rain Rates
    radar_accum['rain_rate_A_accumulated'].plot(x="time", 
                                                ax=ax4,
                                                label="CMAC"
    )

    # PLUVIO2 Accumulation 
    if dis_site == "M1":
        gauge_precip_accum = act.utils.accumulate_precip(ds.sel(station=dis_site), 'intensity_rtnrt').intensity_rtnrt_accumulated.compute()
        gauge_precip_accum.plot(x="time", 
                                ax=ax4,
                                label="PLUVIO2"
        )

    # LDQUANTS Accumulation
    if dis_site == "M1" or dis_site == "S30":
        ld_precip_accum = act.utils.accumulate_precip(ds.sel(station=dis_site), 'rain_rate').rain_rate_accumulated.compute()
        ld_precip_accum.plot(x="time", 
                             ax=ax4,
                             label="LDQUANTS"
        )  
    
    ax4.set_title(" ")
    ax4.set_ylabel("Accumulated Precipitation [mm]")
    ax4.set_xlabel("Time [UTC]")
    ax4.legend(loc="upper left")
    ax4.grid(True)
    ax4.set_ylim(0, radar_accum["rain_rate_A_accumulated"].max()+0.25)
    # Add a blank space next to the subplot to shape it as the above plot
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="3%", pad=1.05)
    cax.set_visible(False)

    # Set the DPI to a higher value (e.g., 300)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # save the figure
    try:
        fig.savefig(out_dir + 
                    'bnf-radclss-.' + 
                    radclss.split('.')[-2]+
                    '.png')
        plt.close(fig)
        STATUS = True
    except:
        STATUS = False

    # Clean up this function 
    del ax2, ax3, ax4
    del gauge_precip_accum, ld_precip_accum, radar_accum
    del ds

    return status

def main(args):
    print(args)
    print("quicklook start time: ", time.strftime("%H:%M:%S"))
    # Define the directories where the CSAPR2 and In-Situ files are located.
    RADCLSS_DIR = args.radclss
    OUT_PATH = args.outdir
    print("\n")
    print("RADCLSS_DIR", RADCLSS_DIR)
    print("OUTPATH: ", OUT_PATH)

    # define the number of days within the month
    d0 = datetime.datetime(year=int(args.month[0:4]), month=int(args.month[4:7]), day=1)
    d1 = datetime.datetime(year=int(args.month[0:4]), month=int(args.month[4:7])+1, day=1)
    volumes = {'date': [], 
               'radclss' : [], 
    }
    
    # Subset dictionary for desired indice 
    def ith_val_subdict(input_dict, i):
        return {k: v[i] for k, v in input_dict.items()}
    
    # iterate through files and collect together
    if args.array is True:
        day_of_month = args.month + args.day
        print("day of month: ", day_of_month)
        volumes['date'].append(day_of_month)
        volumes['radclss'].append(sorted(glob.glob(RADCLSS_DIR + '*' + day_of_month + '*')))
    else:
        for i in range((d1-d0).days):
            if i < 9:
                day_of_month = args.month + '0' + str(i+1)
                print("day of month: ", day_of_month)
                volumes['date'].append(day_of_month)
                volumes['radclss'].append(sorted(glob.glob(RADCLSS_DIR + '*' + day_of_month + '*')))
            else:
                day_of_month = args.month + str(i+1)
                print("day of month: ", day_of_month)
                volumes['date'].append(day_of_month)
                volumes['radclss'].append(sorted(glob.glob(RADCLSS_DIR + '*' + day_of_month + '*')))
    
    if args.verbose:
        print(volumes)
        print("\n")

    # Send volume to RadClss for processing
    for i in range(len(volumes['date'])):
        nvol = ith_val_subdict(volumes, i)
        print(nvol)
        print(nvol["radclss"])
        if nvol["radclss"]:
            status = create_radclss_timeseries(nvol["radclss"])
            #status = radclss(nvol, outdir=OUT_PATH, serial=args.serial, postprocess=args.postproc)
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
                        default='/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradclssS2.c2/quicklooks/',
                        dest='outdir',
                        type=str,
                        help="[str] Specific directory to write RadCLss to"
    )
    
    parser.add_argument("--radclss_dir",
                        default='/gpfs/wolf2/arm/atm124/world-shared/gucxprecipradclssS2.c2/',
                        dest='radclss',
                        type=str,
                        help="[str] Specific directory where the RadCLss files are located"
    )

    parser.add_argument("--station",
                        default="M1",
                        dest="station",
                        type=str,
                        help="[str|default=M1] Specific In-Situ site to display")
   
    parser.add_argument("--verbose",
                        default=False,
                        dest="verbose",
                        type=bool,
                        help="[bool|default=False] Display file paths"
    )
    
    args = parser.parse_args()

    main(args)

#!/usr/bin/env python
#
# Flare mask from University of Utah code with additional cloud, specular reflection, and dark masks. 

# BSD 3-Clause License
#
# Copyright (c) 2019,
#   Scientific Computing and Imaging Institute and
#   Utah Remote Sensing Applications Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Markus Foote (foote@sci.utah.edu)
# Edited to include additional cloud, specular reflection, and dark masks: Andrew Thorpe (Andrew.K.Thorpe@jpl.nasa.gov)

import argparse
import spectral
import numpy as np
#from skimage import morphology, measure
from typing import Tuple, Optional
import matplotlib.pyplot as plt
#import matplotlib as mpl


from scipy import stats
#from scipy.stats import lognorm
import statistics
#import paramnormal
import pandas as pd

import datetime
import math
import os
#from scipy.signal import medfilt
from pyproj import Proj, transform, CRS
from pysolar.solar import *
import boto3
from urllib.parse import urlparse
from os.path import exists

######## FUNCTIONS ########
def get_saturation_mask(data: np.ndarray, wave: np.ndarray, threshold: Optional[float] = None, waverange: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Calculates a mask of pixels that appear saturated (in the SWIR, by default).
    Pixels containing ANY radiance value above the provided threshold (default 6.0) within
    the wavelength window provided (default 1945 - 2485 nm).

    :param data: Radiance image to screen for sensor saturation.
    :param wave: vector of wavelengths (in nanometers) that correspond to the bands (last dimension) in the data.
    Caution: No input validation is performed, so this vector MUST be the same length as the data's last dimension.
    :param threshold: radiance value that defines the edge of saturation.
    :param waverange: wavelength range, defined as a tuple (low, high), to screen within for saturation.
    :return: Binary Mask with 1/True where saturation occurs, 0/False for normal pixels
    """
    if threshold is None:
        threshold = SAT_THRESH_DEFAULT
    if waverange is None:
        waverange = (1945, 2485)
    is_saturated = (data[..., np.logical_and(wave >= waverange[0], wave <= waverange[1])] > threshold).any(axis=-1)
    return is_saturated
   
def get_spec_mask(data: np.ndarray, inp: bool) -> bool:
    """Calculates a mask of pixels that appear to be specular reflection.
    Pixels containing ANY radiance value above the provided threshold at the specified wavelength.
    :param data: Radiance image to screen for sensor saturation.
    :param threshold: radiance values.
    :return: Binary Mask with 1/True where specular reflection occur, 0/False for normal pixels.
    """
    test=data[:, :, 25] # Use radiance data from band 25 
    test2 = test > args.visible_mask_growing_threshold # If high radiance at band 25, could be specular if corresponds to previously identified regions that contain both flares and specular reflection
    is_spec = np.logical_and(inp == 1, test2 == 1) # Define as specular if it was identified in imp (sat_mask_block) and radiance data from band 25 > args.visible_mask_growing_threshold    
    return is_spec        
   
def get_dark_mask(data: np.ndarray) -> bool:
    """Calculates a mask of pixels that are dark.
    Pixels containing ANY radiance value above the provided threshold at the specified wavelength.

    :param data: Radiance image to screen for dark radiance.
    :param wave: vector of wavelengths (in nanometers) that correspond to the bands (last dimension) in the data.
    :param threshold: radiance values.
    :param bandrange: band numbers, defined as a tuple (band_a, band_b, band_c), to screen for clouds.
    :return: Binary Mask with 1/True where clouds occur, 0/False for normal pixels.
    """
    test=data[:, :, 352] # Use radiance data from band 352 (2139 nm)
    test2 = test < args.dark_threshold # If low radiance at band 352, can cause spurious signals (see Ayasse et al., 2018)
    test3 = test <= -9999
    is_spec = np.logical_and(test2 == 1, test3 == 0) # Define as specular if it was identified in imp (sat_mask_block) and radiance data from band 25 > args.visible_mask_growing_threshold    
    return is_spec        
   
def get_cloud_mask(data: np.ndarray, sza, doy):
    """Calculates a mask of pixels that appear to be clouds.
    Pixels containing ANY radiance value above the provided threshold at the specified wavelength.

    :param data: Radiance image to screen for sensor saturation.
    :param sza: solar zenith angle for the image
    :param doy: daoy of year for the image (where Jan 1 is 1 ect).

    :return: Binary Mask with 1/True where clouds occur, 0/False for normal pixels.
    """
    #if threshold is None:
    threshold = (.28, 0.25,0.22) #from Sanford et al. FP 1000, midlat. 
    # other thresholds:
    # Tropics:  (.31, 0.34,0.13)
    # Arctic: (.47, .57, 0.3)
    # Ocean: (.42,.37, .3)
    #if bandrange is None:
    bandrange = (447, 1246, 1650) # AVIRIS-NG bands that will be used based on Thompson et al. 2014, corrsponding to 450 and 1250 nm, 
    # Calculate simple cloud screening based on Thompson et al. 2014 
    rdn1 = ToaRef(bandrange[0],data, sza, doy)
    rdn2 = ToaRef(bandrange[1],data, sza, doy)# this is for snow, not an issue yet
    rdn3 = ToaRef(bandrange[2],data, sza, doy)# this is for snow, not an issue yet
    is_bright447 = rdn1 > threshold[0]
    is_bright1246 = rdn2 > threshold[1]
    is_bright1650 = rdn3 > threshold[2]


    # Combine if the radiance at 450 nm is bright (is_bright) and isNot_snow 
    is_cloud = np.logical_and(  is_bright447[:,:,0] == 1, is_bright1246[:,:,0] == 1, is_bright1650[:,:,0] == 1 )
    
    #return is_bright447
    return is_cloud   



    # Combine if the radiance at 450 nm is bright (is_bright) with negative slopes between band_a and band_b (slope_a_bool) and band_b and band_c (slope_b_bool)
    # If one of the slopes is positive, classify as not a cloud (i.e. bright soil has positive slope between band_a and band_b and neg slope between band_b and band_c)
    is_cloud = np.logical_and(is_bright[:,:,0] == 1, is_hight[:,:,0] == 1)
    
    return is_bright
    #return is_cloud      
   
def get_radius_in_pixels(value_str, metadata):
    if value_str.endswith('px'):
        return np.ceil(float(value_str.split('px')[0]))
    if value_str.endswith('m'):
        if 'map info' not in metadata:
            raise RuntimeError('Image does not have resolution specified. Try giving values in pixels.')
        if 'meters' not in metadata['map info'][10].lower():
            raise RuntimeError('Unknown unit for image resolution.')
        meters_per_pixel_x = float(metadata['map info'][5])
        meters_per_pixel_y = float(metadata['map info'][6])
        if meters_per_pixel_x != meters_per_pixel_y:
            print('Warning: x and y resolutions are not equal, the average resolution will be used.')
            meters_per_pixel_x = (meters_per_pixel_y + meters_per_pixel_x) / 2.0
        pixel_radius = float(value_str.split('m')[0]) / meters_per_pixel_x
        return np.ceil(pixel_radius)
        #raise RuntimeError('Unknown unit specified.')
   
def dilate_mask(binmask, value_str_cld, metadata):
    if value_str_cld.endswith('px'):
        dil_u=np.ceil(float(value_str_cld.split('px')[0])) #Use buffer of this many pixels
    if value_str_cld.endswith('m'):
        if 'map info' not in metadata:
            raise RuntimeError('Image does not have resolution specified. Try giving values in pixels.')
        if 'meters' not in metadata['map info'][10].lower():
            raise RuntimeError('Unknown unit for image resolution.')
        meters_per_pixel_x = float(metadata['map info'][5])
        meters_per_pixel_y = float(metadata['map info'][6])
        if meters_per_pixel_x != meters_per_pixel_y:
            print('Warning: x and y resolutions are not equal, the average resolution will be used.')
            meters_per_pixel_x = (meters_per_pixel_y + meters_per_pixel_x) / 2.0
        dil_u = float(value_str_cld.split('m')[0]) / meters_per_pixel_x #Use buffer of this many pixels based on specified distance
        #raise RuntimeError('Unknown unit specified.')

    from skimage.morphology import binary_dilation as _bwd
    bufmask = binmask.copy()
    for _ in range(int(np.ceil(dil_u))):
        bufmask = _bwd(bufmask)
    return bufmask

def GetSZA(fullfile,linename):
    split_ifile = list(linename)
    first_letter = split_ifile[0]
    
    if first_letter == 'G':
        #Get metadata
        with open(fullfile, 'rb') as f:
            content = f.readlines()
            print(content)
            sam = str(content[3]).split(' ')
            lin = str(content[4]).split(' ')
            map_info = str(content[12]).split(',')
        #print(lin)
        #print(sam)
        #print(map_info)
        
        n_sam = float(str(sam[2]).split('\\')[0])
        n_lin = float(str(lin[2]).split('\\')[0])
        
        dy = float(map_info[5])
        dx = float(map_info[6])
        zone = str(int(map_info[7]))
        easting = float(map_info[3])
        northing = float(map_info[4])
        dx_dy = [dx, dy]
        
        #Get lat/lon coordinates
        utm_x = easting + np.arange(0, (n_sam * dx_dy[1]), dx_dy[1])
        utm_y = northing - np.arange(0, (n_lin * dx_dy[0]), dx_dy[0])
        utm_gridx_nonrotated, utm_gridy_nonrotated = np.meshgrid(utm_x, utm_y)
        
        crs = CRS.from_string('+proj=utm +zone=' + zone + ' +north')
        myProj = Proj(crs.to_authority())
        #myProj = Proj("+proj=utm +zone=" + zone + ", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        slon, slat = myProj(utm_gridx_nonrotated, utm_gridy_nonrotated, inverse=True)
    else:
        #print("there is a problem and this is looking for AVNG data")
        #rdn = spectral.io.envi.open(fullfile)
        with open(fullfile, 'rb') as f:
            content = f.readlines()
            map_info = str(content[11]).split(',')
            sam = str(content[3]).split(' ')
            lin = str(content[4]).split(' ')
        
        n_sam = float(str(sam[2]).split('\\')[0])
        n_lin = float(str(lin[2]).split('\\')[0])
        
    
        dy = float(map_info[6])
        dx = float(map_info[5])
        zone = str(int(map_info[7]))
        easting = float(map_info[3])
        northing = float(map_info[4])
        rot = float(map_info[11].split("=")[1].split('}')[0])
        dx_dy = [dx, dy]
    
        #Get lat/lon coordinates
        utm_x = easting + np.arange(0, (n_sam * dx_dy[1]), dx_dy[1])
        utm_y = northing - np.arange(0, (n_lin * dx_dy[0]), dx_dy[0])
        utm_gridx_nonrotated, utm_gridy_nonrotated = np.meshgrid(utm_x, utm_y)
    
        #Get rotation
        theta = (math.pi/180) * (-1*rot)
        rot_mat = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        def rot_point(theta, x, y):
            xx = x - utm_x[0]
            yy = y - utm_y[0]
    
            new_x = (math.cos(theta) * xx) + (math.sin(theta) * yy)
            new_y = (-math.sin(theta) * xx) + (math.cos(theta) * yy)
    
            return new_x + utm_x[0], new_y + utm_y[0]
    
        utm_grid_x, utm_grid_y = rot_point(theta, utm_gridx_nonrotated, utm_gridy_nonrotated)
    
        crs = CRS.from_string('+proj=utm +zone=' + zone + ' +north')
        myProj = Proj(crs.to_authority())
        #myProj = Proj("+proj=utm +zone=" + zone + ", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        slon, slat = myProj(utm_gridx_nonrotated, utm_gridy_nonrotated, inverse=True)
    
    
    ###### ~~~~~~~ CREATE MATCHED FILTER TARGET SPECTRUM ~~~~~~~ ########
    
    #Get mean lat/lon of scene
    mean_lon = np.mean(slon)
    mean_lat = np.mean(slat)
    
    #Get overpass time
    yy = linename[3:7]
    mm = linename[7:9]
    dd = linename[9:11]
    hh = linename[12:14]
    MM = linename[14:16]
    ss = linename[16:18]
    dtime = datetime.datetime(int(yy), int(mm), int(dd), int(hh), int(MM), int(ss), tzinfo = datetime.timezone.utc)
    
    #Compute solar zenith angle
    SZA = 90 - get_altitude(mean_lat, mean_lon, dtime)
    
    return(SZA)

def GetDOY(linename):
    #Get overpass time
    yy = linename[3:7]
    mm = linename[7:9]
    dd = linename[9:11]
    hh = linename[12:14]
    MM = linename[14:16]
    ss = linename[16:18]
    dtime = datetime.datetime(int(yy), int(mm), int(dd), int(hh), int(MM), int(ss), tzinfo = datetime.timezone.utc)
    day_of_year = dtime.timetuple().tm_yday
    return(day_of_year)

def ToaRef(wvl,data, sza, doy):
    d = (1-0.01672*np.cos(np.radians(0.9856*(doy-4))))# earth sun distance for a given day of the year
    pi = math.pi# pi 
    theta = sza
    wvl = wvl # input wvl
    
    # get the solar illumination
    TAOirad = pd.read_csv('astmg173_read.csv',header=0,index_col=False) #read in TAO dataset from directory
    wvl_arry = TAOirad.Wvlgth.to_numpy() # get wvl as a np array
    irad_arry = TAOirad.DC.to_numpy() #get iradiance as a np arry
    idx = np.argmin(abs(wvl_arry-wvl)) #find the idex that matched the wvl of desire
    s = irad_arry[idx]# solar illumination (using TOA solar irradiance in W/m2/nm)
   
    
    idx2 = np.argmin(abs(np.array(data.bands.centers)- wvl))# get the wvl index from rad data
    y = data[:,:,int(idx2)]*(1e-6)*100**2#original units uW/nm/cm/sr and convert to radiance value in W/nm/sr/m2 
    
    z = ((pi * d**2*y)/(np.cos(np.radians(theta))*(s)))#*y # paper implimentation
    return(z)

def percent_stretch(B):
    B[B<0] = 0
    minval = np.percentile(B, 5)
    maxval = np.percentile(B, 95)
    pixval = (B-minval)/(maxval-minval)
    return pixval

def fetch_from_s3(s3path):
    parsed_url = urlparse(s3path)
    filepath = os.path.basename(parsed_url.path)
    s3 = boto3.client("s3")
    s3.download_file(parsed_url.netloc, parsed_url.path[1:], filepath)
    return filepath
#####################################################################################################
#####################################################################################################
# Define the default radiance value that will be used as a saturation threshold to identify flaring 
SAT_THRESH_DEFAULT = 6.0
# Define the default radiance value that will be used as cloud screening. Values assocuated with 450 and 1250 nm. If you want to use only one threshold, make the other a negative value.
SAT_THRESH_CLD = (0.65, 0.5)
# Define the default radiance value that will be used as dark surface screening at 2139 nm. 
DARK_THRESH_DEFAULT = [0.104]
# Specify default cloud buffer in meters
CLD_BUF='0m'
    
# Define the version of script which will be written to ENVI .hdr file 
SCRIPT_VERSION='1.0.0'

# Parser to permit command line options
parser = argparse.ArgumentParser(description='Flare mask generated for AVIRIS-NG radiance files based on specified radiance threshold for a specified wavelength range.\n' 'f''v{SCRIPT_VERSION}',
                                 epilog='When using this software, please cite: \n' +
                                            ' TBD doi:xxxx.xxxx\n',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False, allow_abbrev=False)
parser.add_argument('--pdf', type=int, nargs=1, default=0,
                    help='Generate pdfs of the rgb and various mask bands to quickly assess performance.')
parser.add_argument('--txt', type=str,
                    help='Text file and file path containing name of files to batch process.')
parser.add_argument('--inpath', type=str,
                    help='File path containing orthocorrected radiance files.')
parser.add_argument('--outpath', type=str,
                    help='File path to write outputs to.')
parser.add_argument('-T', '--saturationthreshold', type=float, metavar='THRESHOLD',
                    help='specify the threshold used for classifying pixels as saturated '
                         'f''(default: {SAT_THRESH_DEFAULT})')
parser.add_argument('-dark', '--dark_threshold', type=float, default=0.104, metavar='FLOAT',
                    help='specify the threshold used for classifying pixels as dark'
                         'f''(default: {DARK_THRESH_DEFAULT})')
parser.add_argument('-C', '--cldthreshold', type=float, nargs=1, default=[15.0],
                    help='specify the threshold used for classifying pixels as saturated '
                         'f''(default: {SAT_THRESH_CLD})')
parser.add_argument('-W', '--saturationwindow', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                    help='specify the contiguous wavelength window within which to detect saturation, independent (default: 1945, 2485 nanometers)')
parser.add_argument('-D', '--cldbands', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                    help='specify the two distinct wavelengths that will be used to detect clouds, independent (default: 450, 1250 nanometers)')
parser.add_argument('-B', '--cldbfr', type=str, metavar='CLDBFR', default='150m',
                    help='specify the cloud buffer distance in meters to mask cloud edges'
                          'f''(default: {CLD_BUF})'),
parser.add_argument('-M', '--maskgrowradius', type=str, metavar='RADIUS', default='150m',
                    help='radius to use for expanding the saturation mask to cover (and exclude) flare-related '
                         'anomalies. This value must include units: meters (abbreviated as m) or pixels '
                         '(abbreviated as px). If flag is given without a value, %(default)s will be used. This is '
                         'a combined flag for enabling mask dilation and setting the distance to dilate.')
parser.add_argument('-A', '--mingrowarea', type=int, metavar='PX_AREA', nargs='?', const=5, default=None,
                    help='minimum number of pixels that must constitute a 2-connected saturation region for it to '
                         'be grown by the mask-grow-radius value. If flag is provided without a value, '
                         '%(const)s pixels will be assumed as the value.')
parser.add_argument('--saturation-processing-block-length', type=int, metavar='N', default=500,
                    help='control the number of data lines pre-processed at once when using masking options')    
parser.add_argument('--visible-mask-growing-threshold', type=float, default=9.0, metavar='FLOAT',
                    help='restrict mask dilation to only occur when 500 nm radiance is less than this value')
parser.add_argument('-o', '--overwrite', action='store_true',
                    help='Force the output files to overwrite any existing files. (default: %(default)s)')
parser.add_argument('-h', '--help', action='help', help='show this help message and exit')
args = parser.parse_args()

print('Arguments:')
print(args)

# Text file path
txt_path = args.txt
# File path containing orthocorrected radiance files
in_path = args.inpath
# File path to write outputs to
out_path = args.outpath

# Read in text file of flights
with open(txt_path, "r") as fd:
  files= fd.read().splitlines()

# Go through each line of the text file 
for f in range(0,len(files)): #Go through each row of text file
    f_txt = str(files[f])
    split_ifile = list(f_txt)
    first_letter = split_ifile[0]
    
    if first_letter == 'G':
        name =''.join(split_ifile[3:11])
        time =''.join(split_ifile[12:18])
        fullfile1 = 's3://carbon-mapper-emit/GAO/2022/rdn/GAO_' + name +'_' + time + '/radiance/' +f_txt + '_rad'
        glt_hdr1 = 's3://carbon-mapper-emit/GAO/2022/rdn/GAO_' + name +'_' + time + '/glt/' +f_txt + '_glt'
        linename = f_txt
        # download and call the GLT file to get the SZA
        file_exists = exists(f_txt + '_rad')
        if file_exists == False:
            fullfile = fetch_from_s3(fullfile1)
            fetch_from_s3(fullfile1 +'.hdr')
        else:
            fullfile = f_txt + '_rad'
        
        file_exists2 = exists(f_txt + '_glt.hdr')
        if file_exists2 == False:
            glt_hdr = fetch_from_s3(glt_hdr1 +'.hdr')
        else:
            glt_hdr = f_txt + '_glt.hdr'
    else:
        fullfile1 = 's3://carbon-mapper-emit/AVIRIS/2022/rdn/' + f_txt + '_rdn_v2aa1_clip'
        glt_hdr1 = 's3://carbon-mapper-emit/AVIRIS/2022/glt/' + f_txt + '_rdn_glt'
        linename = f_txt
        # download and call the GLT file to get the SZA
        file_exists = exists(f_txt + '_rdn_v2aa1_clip')
        if file_exists == False:
            fullfile = fetch_from_s3(fullfile1)
            fetch_from_s3(fullfile1 +'.hdr')
        else:
            fullfile = f_txt + '_rdn_v2aa1_clip'
        
        file_exists2 = exists(f_txt + '_rdn_glt.hdr')
        if file_exists2 == False:
            glt_hdr = fetch_from_s3(glt_hdr1 +'.hdr')
        else:
            glt_hdr = f_txt + '_rdn_glt.hdr'
    print('Processing flight',f_txt)
    # Open the specified radiance file as a memory-mapped object
    rdn_file = spectral.io.envi.open(fullfile + '.hdr',fullfile)
     
    '''
    block_step = args.saturation_processing_block_length
    line_idx_start_values = np.arange(start=0, stop=rdn_file.nrows, step=block_step)

    #this is code for blocking, its a good idea, I am going to comment it out
    
    for line_block_start in line_idx_start_values:
        print('.', end='', flush=True)
        line_block_end = np.minimum(rdn_file.nrows, line_block_start + block_length)
        block_data = rdn_file.read_subregion((line_block_start, line_block_end), (0, rdn_file.ncols))
        sat_mask_block2 = get_cloud_mask(data=block_data[:, :, :], wave=wavelengths,
                                             threshold=args.cldthreshold, bandrange=args.cldbands) # For cloud mask
    '''
    sza = GetSZA(glt_hdr, linename)
    sat_mask_full2 = get_cloud_mask(rdn_file, sza) # For cloud mask
    #Specify output type
    output_dtype = np.int16
    
    # Create an image file for the output
    output_metadata = {'description': 'University of Utah flare and cloud mask.',
                       'band names': ['Cloud mask (dimensionless)','Specular mask (dimensionless)','Flare mask (dimensionless)','Dark mask (dimensionless)'],
                       'interleave': 'bil',
                       'lines': rdn_file.shape[0],
                       'samples': rdn_file.shape[1],
                       'bands': 1,
                       'data type': spectral.io.envi.dtype_to_envi[np.dtype(output_dtype).char]
                       }
    
    # Save results
    
     
    
    output_path = out_path + f_txt[:len('xxxYYYYMMDDtHHMMSS')]  + '_msk_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1')] + '_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_clip')] + '.hdr'
  
    #spectral.envi.save_image(output_path, sat_mask_full2, interleave='bil', ext='', metadata=output_metadata,force=args.overwrite)

    
    output_filename = f_txt[:len('xxxYYYYMMDDtHHMMSS')]  + '_msk_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1')] + '_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_clip')] 
      
    # Save pdfs of rgb and mask bands for evaluation of results
    
    # Plot full scene results to help identify plumes
    # Generate true color image for reference
    rgb = np.zeros((rdn_file.nrows, rdn_file.ncols, 3), dtype=np.float32)
    rgb[:,:,0:1]=percent_stretch(rdn_file[:,:,60])
    rgb[:,:,1:2]=percent_stretch(rdn_file[:,:,42])
    rgb[:,:,2:3]=percent_stretch(rdn_file[:,:,24])
    

    masked_cloud = np.ma.masked_where(sat_mask_full2[:,:] == 0,  sat_mask_full2[:,:])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Cloud Mask')
    ax1.imshow(rgb[:,:,:])
    ax2.imshow(rgb[:,:,:])
    ax2.imshow(masked_cloud,cmap='autumn')
    output_filename_rgb = f_txt[:len('xxxYYYYMMDDtHHMMSS')]  + '_rgb_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1')] + '_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_clip')] 
    plt.savefig(out_path + '/' + output_filename_rgb + '.pdf', bbox_inches = "tight", dpi=500)
    plt.close()
    
        
        
    print('output saved', out_path + '/' + output_filename_rgb + '.pdf')
    print('Generated ' + output_filename)
print('Completed all scenes')


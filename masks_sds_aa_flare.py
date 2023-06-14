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
import os
#from scipy.signal import medfilt

import boto3
from urllib.parse import urlparse
from os.path import exists

######## FUNCTIONS ########
def get_saturation_mask(data: np.ndarray, wave: np.ndarray, threshold: Optional[float] = None, waverange: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Calculates a mask of pixels that appear saturated (in the SWIR, by default).
    Pixels containing ANY radiance value above the provided threshold (default 4.0) within
    the wavelength window provided (default 1945 - 2485 nm).

    :param data: Radiance image to screen for sensor saturation.
    :param wave: vector of wavelengths (in nanometers) that correspond to the bands (last dimension) in the data.
    Caution: No input validation is performed, so this vector MUST be the same length as the data's last dimension.
    :param threshold: radiance value that defines the edge of saturation.
    :param waverange: wavelength range, defined as a tuple (low, high), to screen within for saturation.
    :return: Binary Mask with 1/True where saturation occurs, 0/False for normal pixels
    """
    if threshold is None:
        threshold = 4
    if waverange is None:
        waverange = (2092,2098)
    
    ids = [i for i in range(len(wave)) if wave[i] >= waverange[0] and wave[i]<=waverange[1]]
    
    rdn1 = data[:,:,int(ids[0])]
    rdn2 = data[:,:,int(ids[-1])]
    
    is_bright0 = rdn1 > threshold
    is_bright1 = rdn2 > threshold


    # Combine if the radiance at 450 nm is bright (is_bright) and isNot_snow 
    is_flare = np.logical_and( is_bright0[:,:,0] == 1, is_bright1[:,:,0] == 1)
    
    return is_flare
   
   
  
def dilate_mask(binmask, value_str_cld: Optional[str] = None):
    if value_str_cld is None:
        value_str_cld = '1px'
    
    if value_str_cld.endswith('px'):
        dil_u=np.ceil(float(value_str_cld.split('px')[0])) #Use buffer of this many pixels
    if value_str_cld.endswith('m'):
        raise RuntimeError('does not take m, please give this pixels.')

    from skimage.morphology import binary_dilation as _bwd
    bufmask = binmask.copy()
    for _ in range(int(np.ceil(dil_u))):
        bufmask = _bwd(bufmask)
    return bufmask



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
parser = argparse.ArgumentParser(description='cloud mask generated for AVIRIS-NG radiance files based on specified radiance threshold for a specified wavelength range',

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

parser.add_argument('-T', '--flarethreshold', type=float, metavar='THRESHOLD',
                    help='specify the threshold used for classifying pixels as flares, dfault 4'
                         'f''(default: {SAT_THRESH_DEFAULT})')
parser.add_argument('-W', '--flarewindow', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                    help='specify the contiguous wavelength window within which to detect flare, independent (default: 2090, 2110 nanometers)')
parser.add_argument('-o', '--overwrite', action='store_true',
                    help='Force the output files to overwrite any existing files. (default: %(default)s)')

args = parser.parse_args()

print('Arguments:')
print(args)

# Text file path
txt_path = args.txt
# File path containing orthocorrected radiance files
in_path = args.inpath
# File path to write outputs to
out_path = args.outpath

thresh = args.flarethreshold
#generate pdf arg
pdf = args.pdf

# Read in text file of flights
with open(txt_path, "r") as fd:
  files= fd.read().splitlines()

# Go through each line of the text file 
for f in range(0,len(files)): #Go through each row of text file
#for f in range(0,1: #test 
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
    # This are the saturated pixels (either flare or specular reflection)
    # Get wavelengths from rdn file
    wavelengths = np.array(rdn_file.bands.centers)
    sat_mask_full2 = get_saturation_mask(rdn_file, wave=wavelengths) # For flare mask, pixels saturated in SWIR
    sat_mask_buf = dilate_mask(sat_mask_full2)
   
    sat_mask_buf_int = sat_mask_buf*1
    print(sat_mask_buf_int)
    #Specify output type
    output_dtype = np.int16
    
    # Create an image file for the output
    output_metadata = {'description': 'flare mask.',
                       'band names': ['flare mask (dimensionless)'],
                       'interleave': 'bil',
                       'lines': rdn_file.shape[0],
                       'samples': rdn_file.shape[1],
                       'bands': 1,
                       'data type': spectral.io.envi.dtype_to_envi[np.dtype(output_dtype).char]
                       }
    
    # Save results
     
    output_path = out_path + f_txt[:len('xxxYYYYMMDDtHHMMSS')]  + '_Fmsk_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1')] + '_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_clip')] + '.hdr'
  
    spectral.envi.save_image(output_path, sat_mask_buf_int, interleave='bil', ext='', metadata=output_metadata,force=args.overwrite)

    
    output_filename = f_txt[:len('xxxYYYYMMDDtHHMMSS')]  + '_Fmsk_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1')] + '_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_clip')] 
      
    # Save pdfs of rgb and mask bands for evaluation of results
    if pdf == 0:  
        #do nothing
        print('Generated ' + output_filename)
    else: 
        #Plot full scene results to help identify plumes
        #Generate true color image for reference
        rgb = np.zeros((rdn_file.nrows, rdn_file.ncols, 3), dtype=np.float32)
        rgb[:,:,0:1]=percent_stretch(rdn_file[:,:,60])
        rgb[:,:,1:2]=percent_stretch(rdn_file[:,:,42])
        rgb[:,:,2:3]=percent_stretch(rdn_file[:,:,24])
    

        masked_cloud = np.ma.masked_where(sat_mask_buf[:,:] == 0,  sat_mask_buf  [:,:])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Cloud Mask')
        ax1.imshow(rgb[:,:,:])
        #ax2.imshow(rgb[:,:,:])
        ax2.imshow(masked_cloud,cmap='autumn')
        output_filename_rgb = f_txt[:len('xxxYYYYMMDDtHHMMSS')]  + '_rgb_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1')] + '_' + f_txt[len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_'):len('xxxYYYYMMDDtHHMMSS_rdn_v2x1_clip')] 
        plt.savefig(out_path + '/' + output_filename_rgb + '.pdf', bbox_inches = "tight", dpi=500)
        plt.close()
        print('output saved', out_path + '/' + output_filename + '.pdf')
    
print('Completed all scenes')


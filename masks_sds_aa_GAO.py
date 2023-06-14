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
from skimage import morphology, measure
from typing import Tuple, Optional
from scipy import stats
import statistics
import matplotlib.pyplot as plt
import matplotlib as mpl

########## main functions#############

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
    # Define the default radiance value that will be used as a saturation threshold to identify flaring 
    SAT_THRESH_DEFAULT = 6.0
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

def get_cloud_mask(data: np.ndarray, wave: np.ndarray, bandrange: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Calculates a mask of pixels that appear to be clouds.
    Pixels containing ANY radiance value above the provided threshold at the specified wavelength.

    :param data: Radiance image to screen for sensor saturation.
    :param wave: vector of wavelengths (in nanometers) that correspond to the bands (last dimension) in the data.
    :param threshold: radiance values.
    :param bandrange: band numbers, defined as a tuple (band_a, band_b, band_c), to screen for clouds.
    :return: Binary Mask with 1/True where clouds occur, 0/False for normal pixels.
    """
    if bandrange is None:
        bandrange = (133, 118, 159,139) # AVIRIS-NG bands based on Boa Chi Gaos band ration for clouds. the bands are 1010 nm,940  nm, 1140 nm, 1040 nm  
    
    # Calculate simple cloud screening based on Thompson et al. 2014 
    rdn1010 = data[:, :, bandrange[0]]
    rdn940 = data[:, :, bandrange[1]]
    rdn1140 = data[:, :, bandrange[2]]
    rdn1040 = data[:, :, bandrange[3]]

    
    # normalize the bands              
    rdn_940_n = (rdn940/rdn1010)*1000
    rdn_1140_n = (rdn1140/rdn1010)*1000
    rdn_1040_n = (rdn1040/rdn1010)*1000
    
    
    # Boa Chi Gao band ration
    cloud_BR = (rdn_940_n + rdn_1140_n)/(rdn_1040_n*2)*1000
    
    # get gausian fit
    useidx = lambda Icol: np.where((~(Icol<=0)) & (~(Icol>=1000)))[0]

    cloud_BR_flat = cloud_BR.flatten()
    use = useidx(cloud_BR_flat)
    cloud_BR_flat= cloud_BR_flat[use]
    mu, sigma = stats.norm.fit(cloud_BR_flat)
    median = statistics.median(cloud_BR_flat )
    MAD = stats.median_abs_deviation(cloud_BR_flat)

    if (mu-median)>1:
        thresh =  median+(MAD*2)
    else:
        thresh =  max(cloud_BR_flat)
    
    is_cloud = np.logical_and(cloud_BR >=thresh, cloud_BR <=1000)
    
    return is_cloud    

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


#################### args parse ###########################

# Define the default radiance value that will be used as a saturation threshold to identify flaring 
SAT_THRESH_DEFAULT = 6.0
# Define the default radiance value that will be used as cloud screening. Values assocuated with 450 and 1250 nm. If you want to use only one threshold, make the other a negative value.
SAT_THRESH_CLD = [15.0]
# Define the default radiance value that will be used as dark surface screening at 2139 nm. 
DARK_THRESH_DEFAULT = [0.104]
# Specify default cloud buffer in meters
CLD_BUF='150m'
    
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
    split = str.split(f_txt, '_')
    print(f_txt, split)
    line_name = split[0] + split[1] +'t'+split[2]+'p0000_rad'
    print('Processing flight',line_name)
    # Open the specified radiance file as a memory-mapped object
    rdn_file = spectral.io.envi.open(in_path +'/'+ f_txt + '/'+ 'radiance/' +line_name + '.hdr')
    rdn_file_memmap = rdn_file.open_memmap(interleave='bip', writeable = False)
    # Close memory-mapped object for radiance
    rdn_file_memmap.flush() 
    


    # Get wavelengths from rdn file
    wavelengths = np.array(rdn_file.bands.centers)
    
    # If thresholding is enabled, calculate the mask and (if enabled) preprocess with dilation
    sat_mask_full = None
    print('Detecting pixels to mask...', end='')
    if args.maskgrowradius is not None:
        grow_radius_px = get_radius_in_pixels(args.maskgrowradius, rdn_file.metadata)
        selem = morphology.disk(radius=grow_radius_px, dtype=bool)
        idx_500 = np.argmin(np.absolute(wavelengths - 500))
    sat_mask_full = np.zeros((rdn_file.nrows, rdn_file.ncols), dtype=np.uint8) # For flare mask  
    sat_mask_full2 = np.zeros((rdn_file.nrows, rdn_file.ncols), dtype=np.uint8) # For cloud mask
    dark_mask_full = np.zeros((rdn_file.nrows, rdn_file.ncols), dtype=np.uint8) # For dark mask
    #sat_mask_full3 = np.zeros((rdn_file.nrows, rdn_file.ncols), dtype=np.uint8) # For flare mask buffer
    spec_mask_full = np.zeros((rdn_file.nrows, rdn_file.ncols), dtype=np.uint8) # For specular mask  
    block_overlap = np.ceil((args.mingrowarea if args.mingrowarea is not None else 0) + (grow_radius_px if args.maskgrowradius is not None else 0)).astype(np.int64)
    block_step = args.saturation_processing_block_length
    block_length = block_step + block_overlap
    line_idx_start_values = np.arange(start=0, stop=rdn_file.nrows, step=block_step)
    for line_block_start in line_idx_start_values:
        print('.', end='', flush=True)
        line_block_end = np.minimum(rdn_file.nrows, line_block_start + block_length)
        block_data = rdn_file.read_subregion((line_block_start, line_block_end), (0, rdn_file.ncols))
        sat_mask_block2 = get_cloud_mask(data=block_data[:, :, :], wave=wavelengths,
                                              bandrange=args.cldbands) # For cloud mask
        # This are the saturated pixels (either flare or specular reflection)
        sat_mask_block = get_saturation_mask(data=block_data[:, :, :], wave=wavelengths,
                                             threshold=args.saturationthreshold, waverange=args.saturationwindow) # For flare mask, pixels saturated in SWIR

        # Specify which pixels are specular reflection        
        spec_block = get_spec_mask(data=block_data[:, :, :], inp=sat_mask_block) 
        spec_mask_full[line_block_start:line_block_end, ...][spec_block == 1] = 1 # For specular mask
        sat_mask_full2[line_block_start:line_block_end, ...][sat_mask_block2 == 1] = 1 # For cloud mask
        
        # Apply dark mask
        dark_block = get_dark_mask(data=block_data[:, :, :])
        dark_mask_full[line_block_start:line_block_end, ...][dark_block == 1] = 1 
                
        # Go through flare mask and create an additional mask based on radius in meters
        if args.maskgrowradius is not None:
            sat_mask_grow_regions = np.zeros_like(sat_mask_block, dtype=np.uint8)
            sat_mask_flare = np.zeros_like(sat_mask_block, dtype=np.uint8)
            for region in measure.regionprops(measure.label(sat_mask_block.astype(np.uint8), connectivity=2)):
                if args.mingrowarea is None or region.area >= args.mingrowarea:
                    # Mark these large regions in the mask to get dilated
                    for c in region.coords:
                        # Use visible radiance threshold to rule out sun glint in the visible wavelength range. If sunglint, set mask to 0 (no mask).
                        sat_mask_grow_regions[c[0], c[1]] = 1 if block_data[c[0], c[1], idx_500] < args.visible_mask_growing_threshold else 0 # Define pixels where flare buffer will be applied (this will be a pixel, as opposed to the full flare which would likely contain multoiple pixels)
                        # Binary mask based on radius only for only flares, not for specular reflection
                        sat_mask_large_grown = morphology.binary_dilation(image=sat_mask_grow_regions.astype(bool),
                                                              selem=selem)                  
                        sat_mask_out = sat_mask_large_grown.astype(np.uint8)
                        # Assign flare and specular both 2 (see sat_mask_block), buffer=1
                        sat_mask_out[sat_mask_block] = np.asarray(2, dtype=np.uint8)
                        # Generate a layer of data that will be used as a bankd 
                        sat_mask_full[line_block_start:line_block_end, ...][
                        np.logical_and(sat_mask_large_grown == 1, sat_mask_large_grown == 1)] = 2 # Buffer location assigned to 2 using buffer mask  
                        sat_mask_full[line_block_start:line_block_end, ...][
                        np.logical_and(sat_mask_out == 2, spec_block == 0)] = 1 # Flare location assigned to 1  
    
    # Dialte the cloud mask
        
    #m_pixel = float(rdn_file.metadata['map info'][5])
    #argcldbfr=args.cldbfr
    #cloud_buf_m=np.ceil(float(argcldbfr.split('m')[0]))
    value_str_cld=args.cldbfr
    cloud_mask_buf = dilate_mask(sat_mask_full2, value_str_cld, rdn_file.metadata)

    # Combine the three bands of data
    sat_mask_all = np.zeros((rdn_file.nrows, rdn_file.ncols, 4), dtype=np.int16) 
    sat_mask_all[:,:,0]=cloud_mask_buf # For cloud mask
    sat_mask_all[:,:,1]=spec_mask_full # For specular mask
    sat_mask_all[:,:,2]=sat_mask_full # For flare mask
    sat_mask_all[:,:,3]=dark_mask_full # For flare mask    
    sat_mask_all[rdn_file_memmap[:,:,0]==-9999] = -9999 # Apply the -9999 image border from the radiance file

    #Specify output type
    output_dtype = np.int16
    
    # Create an image file for the output
    output_metadata = {'description': 'University of Utah flare and cloud mask.',
                       'band names': ['Cloud mask (dimensionless)','Specular mask (dimensionless)','Flare mask (dimensionless)','Dark mask (dimensionless)'],
                       'interleave': 'bil',
                       'lines': rdn_file_memmap.shape[0],
                       'samples': rdn_file_memmap.shape[1],
                       'bands': 4,
                       'data type': spectral.io.envi.dtype_to_envi[np.dtype(output_dtype).char],
                       'algorithm settings': '{' f'version: {SCRIPT_VERSION}, ' +
                       (f'flare wvl window: {args.saturationwindow if args.saturationwindow is not None else (1945, 2485)}, ' if args.maskgrowradius else '') +
                       (f'flare threshold: {args.saturationthreshold if args.saturationthreshold is not None else SAT_THRESH_DEFAULT}, ' if args.maskgrowradius else '') +
                       (f'cloud wvl: {args.cldbands if args.cldbands is not None else (450)}, ' if args.maskgrowradius else '') +
                       (f'cloud threshold: {args.cldthreshold if args.cldthreshold is not None else SAT_THRESH_CLD}, ' if args.maskgrowradius else '') +
                       (f'cloud buffer distance: {args.cldbfr if args.cldbfr is not None else 0}, ' if args.maskgrowradius else '') +
                       (f'flare buffer distance: {args.maskgrowradius if args.maskgrowradius is not None else 0}, ' if args.maskgrowradius else '') +
                       (f'flare min contiguous px for buffer: {args.mingrowarea if args.mingrowarea is not None else 0}, ' if args.maskgrowradius else '') +
                       (f'500 nm mask buffering threshold: {args.visible_mask_growing_threshold}, ' if args.maskgrowradius else '') +
                       (f'darkthreshold: {args.dark_threshold}, ' if args.maskgrowradius else '') +
                       f'parsed cmdline args: {args}'
                       '}'}
    
    # Save results
    
    if value_str_cld.endswith('m'):
        output_metadata.update({'map info': rdn_file.metadata['map info']})
    
    if value_str_cld.endswith('m'):
        output_path = out_path + line_name[:len('xxxYYYYMMDDtHHMMSS')]  + '_msk'  + '.hdr'
    else:
        output_path = out_path + line_name[:len('xxxYYYYMMDDtHHMMSS')]  + '_msk'  + '.hdr'
  
    spectral.envi.save_image(output_path, sat_mask_all, interleave='bil', ext='', metadata=output_metadata,force=args.overwrite)

    if value_str_cld.endswith('m'):    
        output_filename = line_name[:len('xxxYYYYMMDDtHHMMSS')]  + '_msk'
    else:
        output_filename = line_name[:len('xxxYYYYMMDDtHHMMSS')]  + '_msk'
    
#    # Save pdfs of rgb and mask bands for evaluation of results
    if args.pdf !=0:
        def percent_stretch(B):
           B[B<0] = 0
           minval = np.percentile(B, 5)
           maxval = np.percentile(B, 95)
           pixval = (B-minval)/(maxval-minval)
           return pixval
        # Plot full scene results to help identify plumes
       
       # Generate true color image for reference
        rgb = np.zeros((rdn_file.nrows, rdn_file.ncols, 3), dtype=np.float32)
        
        r = rdn_file[:,:,59]
        g = rdn_file[:,:,38]
        b = rdn_file[:,:,16]
        rgb[:,:,0:1]= percent_stretch(r)#59
        rgb[:,:,1:2]= percent_stretch(g)#38
        rgb[:,:,2:3]= percent_stretch(b)#16
        
#        
#        # Replace -9999 with 0
        rgb2=np.where(rgb==-9999, 0, rgb) 
#    

#    
        size=sat_mask_full.shape
        line=size[0]
        samp=size[1]
#        
        png_dpi=500
#        
        asp = line/samp*0.6
        fig_x_in = 10
        fig_y_in = fig_x_in*asp
#        
        fig, ax = plt.subplots(1)
        fig.set_figheight(fig_y_in)
        fig.set_figwidth(fig_x_in)
        view = ax.imshow(rgb)
        ax.set_xticks(np.arange(0, samp, 100))
        ax.set_yticks(np.arange(0, line, 100))
        ax.set_xticklabels(np.arange(0, samp+1, 100))
        ax.set_yticklabels(np.arange(0, line+1, 100))
        ax.grid(color = 'w', linestyle = '-', linewidth = 1)
        output_filename_rgb = line_name[:len('xxxYYYYMMDDtHHMMSS')]  + '_rgb' 
        plt.savefig(out_path + '/' + output_filename_rgb + '.pdf', bbox_inches = "tight", dpi=png_dpi)
        plt.close()
#    
#        # Plot cloud mask (buffered)
        par_cmap = mpl.colors.ListedColormap(['black', 'white'])
        bounds = [0, 1]
#        
        fig, ax = plt.subplots(1)
        fig.set_figheight(fig_y_in)
        fig.set_figwidth(fig_x_in)
        view = ax.imshow(cloud_mask_buf, cmap=par_cmap)
        #cb=fig.colorbar(view, ax=ax1)
        #cb.set_label('ppm-m')
        ax.set_xticks(np.arange(0, samp, 100))
        ax.set_yticks(np.arange(0, line, 100))
        ax.set_xticklabels(np.arange(0, samp+1, 100))
        ax.set_yticklabels(np.arange(0, line+1, 100))
        ax.grid(color = 'w', linestyle = '-', linewidth = 1)
        plt.savefig(out_path + '/' + output_filename + '_b1_cloud.pdf', bbox_inches = "tight", dpi=png_dpi)
        plt.close()
#    
#        # Plot specular mask            
        par_cmap = mpl.colors.ListedColormap(['black', 'yellow'])
        bounds = [0, 1]
#    
        fig, ax = plt.subplots(1)
        fig.set_figheight(fig_y_in)
        fig.set_figwidth(fig_x_in)
        view = ax.imshow(spec_mask_full, cmap=par_cmap)
        #cb=fig.colorbar(view, ax=ax1)
        #cb.set_label('ppm-m')
        ax.set_xticks(np.arange(0, samp, 100))
        ax.set_yticks(np.arange(0, line, 100))
        ax.set_xticklabels(np.arange(0, samp+1, 100))
        ax.set_yticklabels(np.arange(0, line+1, 100))
        ax.grid(color = 'w', linestyle = '-', linewidth = 1)
        plt.savefig(out_path + '/' + output_filename + '_b2_specular.pdf', bbox_inches = "tight", dpi=png_dpi)
        plt.close()
#    
#        # Plot flare mask            
        par_cmap = mpl.colors.ListedColormap(['black', 'red', 'green'])
        bounds = [0, 1, 2]
#    
        fig, ax = plt.subplots(1)
        fig.set_figheight(fig_y_in)
        fig.set_figwidth(fig_x_in)
        view = ax.imshow(sat_mask_full, cmap=par_cmap)
        #cb=fig.colorbar(view, ax=ax1)
        #cb.set_label('ppm-m')
        ax.set_xticks(np.arange(0, samp, 100))
        ax.set_yticks(np.arange(0, line, 100))
        ax.set_xticklabels(np.arange(0, samp+1, 100))
        ax.set_yticklabels(np.arange(0, line+1, 100))
        ax.grid(color = 'w', linestyle = '-', linewidth = 1)
        plt.savefig(out_path + '/' + output_filename + '_b3_flare.pdf', bbox_inches = "tight", dpi=png_dpi)
        plt.close()
#        
#        # Plot dark mask            
        par_cmap = mpl.colors.ListedColormap(['black', 'orange'])
        bounds = [0, 1]
#    
        fig, ax = plt.subplots(1)
        fig.set_figheight(fig_y_in)
        fig.set_figwidth(fig_x_in)
        view = ax.imshow(dark_mask_full, cmap=par_cmap)
        #cb=fig.colorbar(view, ax=ax1)
        #cb.set_label('ppm-m')
        ax.set_xticks(np.arange(0, samp, 100))
        ax.set_yticks(np.arange(0, line, 100))
        ax.set_xticklabels(np.arange(0, samp+1, 100))
        ax.set_yticklabels(np.arange(0, line+1, 100))
        ax.grid(color = 'w', linestyle = '-', linewidth = 1)
        plt.savefig(out_path + '/' + output_filename + '_b4_dark.pdf', bbox_inches = "tight", dpi=png_dpi)
        plt.close()
#        
#
    print('Generated ' + output_filename)

print('Completed all scenes')


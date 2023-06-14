# files the matter
There are currently two scrips that matter here. 
masks_sds_aa_flare.py and maska_sds_aa_clouds.py
these scripts are the flare and cloud mask respectivly, they shouls operate the same way.

# required flags
Specify text file with list of radiance files to process.\
--txt 

Specify file path for folder where mask results will be saved.\
--outpath

Right now the input folders are hard coded because I needed to harcode s3 directories, this can be changed.\
currently this code reads from S3 and writes the files to the directory where this code is. I havent figured out \
how to change this yet and we will need to work with Justin on this issue. 

All other flags are optional 

# Running on non-ortho scene:
python masks_sds_aa_flare.py --txt /path/input_list.txt --inpath /path/input_nonortho_rad/ --outpath /path/saved_flare_mask/ 

currently this code is only running on non-ortho scenes but we can change that. 

# environment
there is a efs conda environment that works for this. 
/efs/conda/envmasks



###################### Old README from Marcus ########################################
# Flare mask from University of Utah code with additional cloud, specular reflection, and dark masks

A four band ENVI file is generated that contains the following:\
Band 1, cloud mask (0=no mask, 1=cloud mask). Uses radiance threshold, see get_cloud_mask.\
Band 2, specular reflection (0=no mask, 1=specular reflection mask). Uses radiance threshold see get_spec_mask.\
Band 3, flare mask (0=no mask, 1=flare detection mask, 2=flare buffer  mask). Uses radiance threshold, see get_saturation_mask.\
Band 4, dark mask (0=no mask, 1=dark mask). Uses radiance threshold see get_dark_mask.


# Citation
If you use this tool in a program or publication, please acknowledge its author(s) by adding the following reference: TBD



# Runtime Options
There are numerous options/flags that can be provided to modify processing behavior. Run `masks --help` for a full description of the available arguments.

## Required flags are:

Specify text file with list of radiance files to process.\
--txt

Specify file path for folder containing radiance files.\
--inpath

Specify file path for folder where mask results will be saved.\
--outpath

Specifiy radius of circular radiance mask. This can be in meters for orthocorrected results (i.e. 150m) or in pixels for non-orthocorrected results (i.e. 75px).\
--maskgrowradius 150m

Specifiy dilation distance that will be applied to cloud mask. This can be in meters for orthocorrected results (i.e. 150m, which is a good setting) or in pixels for non-orthocorrected results (i.e. 25px, which is a good setting).\
--cldbfr 150m


## Additional imporant flags:

Specifiy flare saturation radiance threshold (default is 15).\
--saturationthreshold 15

Specifiy dark radiance threshold (default is 0.104).\
--dark_threshold 0.104

Specifiy cloud radiance threshold (default is 15).\
--cldthreshold 15
 
Specify the contiguous wavelength window within which to detect saturation (default is 1945, 2485).\
--saturationwindow 1945 2485

Specify the three bands used for cloud detection (default is 15, 60, 175). See get_cloud_mask.\
--cldbands

Specify radiance threshold for specular reflection mask defined by 500 nm radiance (default is 9).\
--visible-mask-growing-threshold 10

Specify minimum number of pixels that must constitute a 2-connected saturation region for it to be grown by the mask-grow-radius value (dafault 5).\
--mingrowarea 10

Specify control the number of data lines pre-processed at once when using masking options (default is 500).\
--saturation-processing-block-length 500

Specify restrict mask dilation to only occur when 500 nm radiance is less than this value (default is 9). This is used for the specular reflection mask (see get_spec_mask).\
--visible-mask-growing-threshold 9

Specifiy if you want to overwrite previous files.\
--overwrite

Generate pdfs of the rgb and various mask bands to quickly assess performance (0=no pdf, 1=generate pdfs).\
--pdf 1


# Examples
## Process a set of files:

### Running on ortho scene:
python masks.py --txt /path/input_list.txt --inpath /path/input_ortho_rad/ --outpath /path/saved_flare_mask/ --maskgrowradius 150m --cldbfr 150m

### Running on non-ortho scene:
python masks.py --txt /path/input_list.txt --inpath /path/input_nonortho_rad/ --outpath /path/saved_flare_mask/ --maskgrowradius 25px --cldbfr 25px

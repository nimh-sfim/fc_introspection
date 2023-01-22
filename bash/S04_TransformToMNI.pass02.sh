set -e

cd /data/SFIMJGC_Introspec/pdn/PrcsData

# Create New Folder where we will generate new grid
if [ ! -d ALL_SCANS ]; then mkdir ALL_SCANS; fi

# Enter target folder
cd ALL_SCANS

# Create the averate across all scans that enter our analysis (471 scans)
3dMean      -overwrite -prefix all_mean.nii.gz `ls ../sub-??????/preprocessed/func/pb05_mni/_scan_id_ses-02_task-rest_acq-??_run-??_bold/rest_mean_2mni.nii.gz`

# Remove the skull
3dAutomask  -overwrite -prefix all_mean.SS.nii.gz all_mean.nii.gz

# Dilate (just to make sure we are not unnecessarily removing edge voxels of interest)
3dmask_tool -overwrite -prefix all_mean.mask.nii.gz -dilate_input 1 -input all_mean.SS.nii.gz

# Fill holes (to make sure we do not remove any internal structures at the bottom of the brain)
3dmask_tool -overwrite -prefix all_mean.mask.nii.gz -fill_holes -fill_dirs xy -input all_mean.mask.nii.gz

# Generate the new smaller grid
3dAutobox   -overwrite -prefix all_mean.box.nii.gz  all_mean.mask.nii.gz

# Ensure all generate files have the space flag set to MNI
3drefit -space MNI all_mean.box.nii.gz all_mean.mask.nii.gz all_mean.nii.gz all_mean.SS.nii.gz

# Bring Files of interest in this folder to smaller grid
3dZeropad -overwrite -master all_mean.box.nii.gz -prefix all_mean.boxed.nii.gz all_mean.nii.gz
3dZeropad -overwrite -master all_mean.box.nii.gz -prefix all_mean.SS.boxed.nii.gz all_mean.SS.nii.gz
3dZeropad -overwrite -master all_mean.box.nii.gz -prefix all_mean.mask.boxed.nii.gz all_mean.mask.nii.gz

# Print completion message
echo "++ INFO: Script finished correctly"
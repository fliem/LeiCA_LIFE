import os
from nilearn import datasets
import nibabel as nb
import numpy as np
import shutil

atlas = datasets.fetch_atlas_craddock_2012(data_dir='.')

atlas_str = 'scorr_mean'

cc_file = atlas[atlas_str]
cc_folder = os.path.split(cc_file)[0]

save_path = os.path.join(cc_folder, atlas_str + '_single_resolution')
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

cc_4d_img = nb.load(cc_file)
cc_4d_data = cc_4d_img.get_data()

n_resolution = cc_4d_data.shape[-1]
n_parcels = [np.unique(cc_4d_data[:, :, :, i]).shape[0] - 1 for i in
             range(n_resolution)]  # -1 to account for 0 background

for parc_to_use in range(n_resolution):
    print 'using %s parcels' % n_parcels[parc_to_use]

    # save 3d nii of desired parc.resolution
    cc_3d_filename = os.path.join(save_path,
                                  atlas_str + '_parc_n_%s_k_%s_rois.nii.gz' % (parc_to_use+1, n_parcels[parc_to_use]))
    cc_3d_img = nb.Nifti1Image(cc_4d_data[:, :, :, parc_to_use], cc_4d_img.get_affine(), cc_4d_img.get_header())
    cc_3d_img.to_filename(cc_3d_filename)
    print 'save to %s' % cc_3d_filename

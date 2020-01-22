'''
wd ca 170MB/subj
'''
import os
from metrics.calc_metrics_scrubbing import calc_local_metrics
from variables import template_dir, subjects_list, metrics_root_path, wd_root_path


working_dir_base = os.path.join(wd_root_path, 'wd_metrics_scrubbing')
in_data_root_path = os.path.join(metrics_root_path, 'metrics')
ds_dir_base = os.path.join(metrics_root_path, 'metrics')

brain_mask = 'PATH/Templates/MNI_resampled_brain_mask.nii'

template_dir = os.path.join(template_dir, 'parcellations')

fd_thresh = .5
bp_freq_list = ['0.01.0.1']

parcellations_dict = {}
parcellations_dict['craddock_205'] = {
    'nii_path': os.path.join(template_dir,
                             'craddock_2012/scorr_mean_single_resolution/scorr_mean_parc_n_21_k_205_rois.nii.gz'),
    'is_probabilistic': False}
parcellations_dict['craddock_788'] = {
    'nii_path': os.path.join(template_dir,
                             'craddock_2012/scorr_mean_single_resolution/scorr_mean_parc_n_43_k_788_rois.nii.gz'),
    'is_probabilistic': False}
parcellations_dict['gordon'] = {
    'nii_path': os.path.join(template_dir, 'Gordon_2014_Parcels/Parcels_MNI_111_sorted.nii.gz'),
    'is_probabilistic': False}

use_n_procs = 5
# plugin_name = 'MultiProc'
plugin_name = 'CondorDAGMan'

for subject_id in subjects_list:
    working_dir = os.path.join(working_dir_base, subject_id)
    ds_dir = os.path.join(ds_dir_base, subject_id)

    print('\n\nsubmitting %s' % subject_id)
    calc_local_metrics(preprocessed_data_dir=in_data_root_path,
                       subject_id=subject_id,
                       parcellations_dict=parcellations_dict,
                       bp_freq_list=bp_freq_list,
                       fd_thresh=fd_thresh,
                       working_dir=working_dir,
                       ds_dir=ds_dir,
                       use_n_procs=use_n_procs,
                       plugin_name=plugin_name)

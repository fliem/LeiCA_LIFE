'''
wd ca 170MB/subj
'''
import os
from metrics.calc_metrics import calc_local_metrics
from variables import template_dir, in_data_root_path, subjects_list, \
    metrics_root_path, wd_root_path, selectfiles_templates

working_dir_base = os.path.join(wd_root_path, 'wd_metrics')
ds_dir_base = os.path.join(metrics_root_path, 'metrics_test')

brain_mask = 'PATH/Templates/MNI_resampled_brain_mask.nii'

template_dir = os.path.join(template_dir, 'parcellations')

# con mat parameters
bp_freq_list =  [(0.01, 0.1)] #[(None, None), (0.01, 0.1)]
TR = 2.0

parcellations_dict = {}

parcellations_dict['basc_444'] = {
    'nii_path': os.path.join(template_dir,
                             'basc_multiscale_2015/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale444.nii.gz'),
    'is_probabilistic': False}
parcellations_dict['basc_197'] = {
    'nii_path': os.path.join(template_dir,
                             'basc_multiscale_2015/template_cambridge_basc_multiscale_nii_sym/template_cambridge_basc_multiscale_sym_scale197.nii.gz'),
    'is_probabilistic': False}

use_n_procs = 5
# plugin_name = 'MultiProc'
plugin_name = 'CondorDAGMan'

for subject_id in subjects_list:
    working_dir = os.path.join(working_dir_base, subject_id)
    ds_dir = os.path.join(ds_dir_base, subject_id)

    print('\n\nsubmitting %s' % subject_id)
    calc_local_metrics(brain_mask=brain_mask,
                       preprocessed_data_dir=in_data_root_path,
                       subject_id=subject_id,
                       parcellations_dict=parcellations_dict,
                       bp_freq_list=bp_freq_list,
                       TR=TR,
                       selectfiles_templates=selectfiles_templates,
                       working_dir=working_dir,
                       ds_dir=ds_dir,
                       use_n_procs=use_n_procs,
                       plugin_name=plugin_name)

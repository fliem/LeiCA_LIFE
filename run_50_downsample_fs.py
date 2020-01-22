'''
'''
import os
from freesurfer.downsample_surfs import downsampel_surfs
from variables import subjects_list, metrics_root_path, wd_root_path
from utils import load_subjects_list

#fixme
subject_file = 'PATH/redo_fs.txt'
subjects_list = load_subjects_list(subject_file)


working_dir_base = os.path.join(wd_root_path, 'wd_fs')
ds_dir_base = os.path.join(metrics_root_path, 'metrics')

fresurfer_dir = 'PATH/freesurfer_all'


use_n_procs = 1
#plugin_name = 'MultiProc'
plugin_name = 'CondorDAGMan'


for subject_id in subjects_list:
    working_dir = os.path.join(working_dir_base, subject_id)
    ds_dir = os.path.join(ds_dir_base, subject_id)
    print('\nsubmitting %s'%subject_id)
    downsampel_surfs(subject_id=subject_id,
                     working_dir=working_dir,
                     freesurfer_dir=fresurfer_dir,
                     ds_dir=ds_dir,
                     plugin_name=plugin_name,
                     use_n_procs=use_n_procs)

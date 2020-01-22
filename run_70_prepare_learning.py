import os
import pandas as pd

# # LeiCA modules
from learning.learning_prepare_data_wf import learning_prepare_data_wf
from learning.learning_variables import in_data_name_list
from variables import wd_root_path, ds_root_path, data_lookup_dict, template_lookup_dict
from variables import behav_file, qc_file

wd_root_path = 'PATH/LeiCA_LIFE_learning_wd'
ds_root_path = 'PATH/LeiCA_LIFE'

working_dir = os.path.join(wd_root_path, 'wd_learning')
ds_dir = os.path.join(ds_root_path, 'learning_out')

use_n_procs = 1
plugin_name = 'MultiProc'

learning_prepare_data_wf(working_dir=working_dir,
                         ds_dir=ds_dir,
                         template_lookup_dict=template_lookup_dict,
                         behav_file=behav_file,
                         qc_file=qc_file,
                         in_data_name_list=in_data_name_list,
                         data_lookup_dict=data_lookup_dict,
                         use_n_procs=use_n_procs,
                         plugin_name=plugin_name)

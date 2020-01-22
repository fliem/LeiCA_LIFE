import os

# # LeiCA modules
from learning.learning_predict_data_wf import learning_predict_data_2samp_wf
from learning.learning_variables import in_data_name_list, subjects_selection_crit_dict, \
    subjects_selection_crit_names_list, target_list



use_n_procs = 6
plugin_name = 'MultiProc'


working_dir = 'PATH'
ds_dir = 'PATH'
aggregated_subjects_dir = 'PATH'

aggregated_subjects_dir_nki = 'PATH'
subjects_selection_crit_dict_nki = {}
subjects_selection_crit_dict_nki['adult'] = ["age >= 18", "n_TRs > 890"]
subjects_selection_crit_name_nki = 'adult'

# LIFE only training
run_2sample_training = False
learning_predict_data_2samp_wf(working_dir=working_dir,
                               ds_dir=ds_dir,
                               in_data_name_list=in_data_name_list,
                               subjects_selection_crit_dict=subjects_selection_crit_dict,
                               subjects_selection_crit_names_list=subjects_selection_crit_names_list,
                               aggregated_subjects_dir=aggregated_subjects_dir,
                               target_list=target_list,
                               use_n_procs=use_n_procs,
                               plugin_name=plugin_name,
                               confound_regression=[False, True],
                               run_cv=True,
                               n_jobs_cv=5,
                               run_tuning=False,
                               run_2sample_training=run_2sample_training,
                               aggregated_subjects_dir_nki=aggregated_subjects_dir_nki,
                               subjects_selection_crit_dict_nki=subjects_selection_crit_dict_nki,
                               subjects_selection_crit_name_nki=subjects_selection_crit_name_nki)




# LIFE + NKI training
working_dir = 'PATH'
ds_dir = 'PATH'
run_2sample_training = True
learning_predict_data_2samp_wf(working_dir=working_dir,
                               ds_dir=ds_dir,
                               in_data_name_list=in_data_name_list,
                               subjects_selection_crit_dict=subjects_selection_crit_dict,
                               subjects_selection_crit_names_list=subjects_selection_crit_names_list,
                               aggregated_subjects_dir=aggregated_subjects_dir,
                               target_list=target_list,
                               use_n_procs=use_n_procs,
                               plugin_name=plugin_name,
                               confound_regression=[False, True],
                               run_cv=True,
                               n_jobs_cv=5,
                               run_tuning=False,
                               run_2sample_training=run_2sample_training,
                               aggregated_subjects_dir_nki=aggregated_subjects_dir_nki,
                               subjects_selection_crit_dict_nki=subjects_selection_crit_dict_nki,
                               subjects_selection_crit_name_nki=subjects_selection_crit_name_nki)

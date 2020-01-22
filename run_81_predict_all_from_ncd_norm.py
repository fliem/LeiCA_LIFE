import os
import pandas as pd

# # LeiCA modules
from LeiCA_LIFE.learning.learning_predict_data_from_trained_model_wf import learning_predict_data_wf

from learning.learning_variables import in_data_name_list, subjects_selection_crit_dict, target_list



use_n_procs = 6
plugin_name = 'MultiProc'

aggregated_subjects_dir = os.path.join("PATH", 'vectorized_aggregated_data')


train_subjects_selection_crit_list = ['bothSexes_neuH_FD06_ncd_norm']
subjects_selection_crit_names_list =['bothSexes_neuH_FD06']

for training in ['training_life_only']:
    ds_root_path = os.path.join("PATH", training)
    working_dir = os.path.join("PATH", training)
    trained_model_dir = os.path.join("PATH", training)

    for train_subjects_selection_crit in train_subjects_selection_crit_list:
        ds_dir = os.path.join(ds_root_path, train_subjects_selection_crit)
        trained_model_template = {
            'trained_model': 'group_learning_prepare_data/{ana_stream}trained_model/' +
                             '_multimodal_in_data_name_{multimodal_in_data_name}/_selection_criterium_' + train_subjects_selection_crit + '/' +
                             '_target_name_{target_name}/trained_model.pkl'}

        learning_predict_data_wf(working_dir=working_dir,
                                 ds_dir=ds_dir,
                                 trained_model_dir=trained_model_dir,
                                 in_data_name_list=in_data_name_list,
                                 subjects_selection_crit_dict=subjects_selection_crit_dict,
                                 subjects_selection_crit_names_list=subjects_selection_crit_names_list,
                                 aggregated_subjects_dir=aggregated_subjects_dir,
                                 target_list=target_list,
                                 trained_model_template=trained_model_template,
                                 use_n_procs=use_n_procs,
                                 plugin_name=plugin_name,
                                 confound_regression=[False])

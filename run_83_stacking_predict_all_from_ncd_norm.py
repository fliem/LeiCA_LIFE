import os
from learning.learning_stacking_utils import stacking

date_str = '20160504'

target = 'age'
selection_crit_test = 'bothSexes_neuH_FD06'


in_data_path = "PATH"

selection_crit_nki = 'bothSexes_neuH_FD06'

life_subjects_selection_crit_list = ['bothSexes_neuH_FD06_ncd_norm']


for training in ['training_life_only']:
    root_path_template = os.path.join(in_data_path,
                                      'learning_out_predict_all_from_ncd_norm_' + date_str + '/' + training + '/{life_subjects_selection_crit}/pdfs/single_source_model_reg_{reg}_predicted')
    out_path_template = os.path.join(in_data_path,
                                     'learning_out_predict_all_from_ncd_norm_' + date_str + '/' + training + '/{life_subjects_selection_crit}/stacking/stacking_out_reg_{reg}')
    rf_root_template = os.path.join(in_data_path,
                                    'learning_out_ncd_norm_' + date_str + '/' + training + '/stacking/stacking_out_reg_{reg}')

    for life_subjects_selection_crit in life_subjects_selection_crit_list:
        rf_file_template = target + '__' + life_subjects_selection_crit + '__{stacking_crit}__stacking_fitted_model.pkl'

        for reg in [False]:
            root_path = root_path_template.format(reg=reg, life_subjects_selection_crit=life_subjects_selection_crit)
            out_path = out_path_template.format(reg=reg, life_subjects_selection_crit=life_subjects_selection_crit)

            file_pref = target + '__' + selection_crit_nki + '__'
            source_dict = {
                'aseg': os.path.join(root_path, file_pref + 'aseg_df_predicted.pkl'),
                'ct': os.path.join(root_path, file_pref + 'lh_ct_fsav4_sm0__rh_ct_fsav4_sm0_df_predicted.pkl'),
                'csa': os.path.join(root_path, file_pref + 'lh_csa_fsav4_sm0__rh_csa_fsav4_sm0_df_predicted.pkl'),
                'basc197': os.path.join(root_path, file_pref + 'basc_197_df_predicted.pkl'),
                'basc444': os.path.join(root_path, file_pref + 'basc_444_df_predicted.pkl'),
            }

            source_selection_dict = {'all': ['basc197', 'basc444', 'aseg', 'csa', 'ct'],
                                     'rs': ['basc197', 'basc444'],
                                     'fs': ['aseg', 'csa', 'ct'],
                                     }

            rf_file = os.path.join(rf_root_template.format(reg=reg), rf_file_template)
            stacking(out_path, target, life_subjects_selection_crit, source_dict, source_selection_dict, rf=rf_file)

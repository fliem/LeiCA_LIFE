import os
from learning.learning_stacking_utils import stacking


target = 'age'
selection_crit_list = ['bothSexes_neuH_FD06']

date_str = '20160504'

in_data_path="PATH"

for training in ['training_life_only']:
    print(training)
    root_path_template = os.path.join(in_data_path,'learning_out_reverse_split_' + date_str, training,'pdfs/single_source_model_reg_{reg}_predicted')
    out_path_template =  os.path.join(in_data_path, 'learning_out_reverse_split_' + date_str, training,'stacking/stacking_out_reg_{reg}')

    for reg in [False]:
        for selection_crit in selection_crit_list:
            root_path = root_path_template.format(reg=reg)
            out_path = out_path_template.format(reg=reg)

            file_pref = target + '__' + selection_crit + '__'
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



            stacking(out_path, target, selection_crit, source_dict, source_selection_dict, rf=None)

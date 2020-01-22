# # # # # # # # # #
# subject selection criteria
# # # # # # # # # #
subjects_selection_crit_dict = {}
# d['a'] = ["a < 3", "b > 2"]
# NEW Style!
subjects_selection_crit_dict['bothSexes'] = ["n_TRs > 294"]
subjects_selection_crit_dict['bothSexes_neuH'] = ["n_TRs > 294", "neurol_healthy==True"]
subjects_selection_crit_dict['bothSexes_neuH_FD06'] = ["n_TRs > 294", "neurol_healthy==True", "mean_FD_P<0.6"]
subjects_selection_crit_dict['bothSexes_neuH_FD06_ncd_norm'] = ["n_TRs > 294", "neurol_healthy==True", "mean_FD_P<0.6", "ncd_group=='norm'"]
subjects_selection_crit_dict['bothSexes_FD06'] = ["n_TRs > 294", "mean_FD_P<0.6"]
subjects_selection_crit_dict['bothSexes_neuH_FD06_norm'] = ["n_TRs > 294", "neurol_healthy==True", "mean_FD_P<0.6",
                                                            "cs_mean_group_int == 0"] # equiv to cs_mean_group == 'norm'
subjects_selection_crit_dict['bothSexes_neuH_FD06_nonorm'] = ["n_TRs > 294", "neurol_healthy==True", "mean_FD_P<0.6",
                                                            "cs_mean_group_int > 0"]
# in_data_name_list = [
#     ['dosenbach'],
#     ['basc_197'],
#     # ['basc_444'],
#     ['craddock_205_BP'],
#     # ['craddock_788_BP'],
#     # ['craddock_205_BP_scr05'],
#     # ['craddock_788_BP_scr05'],
#     # ['msdl_abide_BP'],
#
#     ['lh_ct_fsav4_sm0', 'rh_ct_fsav4_sm0'],
#     ['lh_csa_fsav4_sm0', 'rh_csa_fsav4_sm0'],
#     ['aseg'],
#     ['aparc_lh_thickness', 'aparc_rh_thickness'],
#     ['aparc_rh_area', 'aparc_lh_area'],
#
#     ['behav_wml_wm_total'], ['behav_wml_wmh_norm'], ['behav_wml_wmh_norm_ln'],
#     ['behav_wml_wmh_ln'],
#     ['behav_wml_fazekas'], ['behav_wml_wmh_norm_ln', 'behav_wml_fazekas'],
# ]
in_data_name_list = [
    ['basc_197'],
    ['basc_444'],
    ['aseg'],
    ['lh_ct_fsav4_sm0', 'rh_ct_fsav4_sm0'],
    ['lh_csa_fsav4_sm0', 'rh_csa_fsav4_sm0'],

]

# subjects_selection_crit_names_list = ['bothSexes_neuH']
subjects_selection_crit_names_list = ['bothSexes_neuH_FD06'] #, 'bothSexes_neuH_FD06_norm']

target_list = ['age']


# check for duplicates in in_data_file_list
seen = []
for i in in_data_name_list:
    if i in seen:
        raise Exception('duplicate in in_data_name_list: %s' % i)
    else:
        seen.append(i)

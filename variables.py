import os
from utils import load_subjects_list

script_dir = os.path.dirname(os.path.realpath(__file__))
template_dir = os.path.join(script_dir, 'templates')

wd_root_path = "PATH"
metrics_root_path = "PATH"
ds_root_path ="PATH"

subjects_list_folder = "PATH"

subject_file = "PATH"

behav_file = "PATH"
qc_file = "PATH"

subjects_list = load_subjects_list(subject_file)

in_data_root_path = "PATH"

selectfiles_templates = {
    'epi_MNI_bp': '{subject_id}/resting_state/ants/rest_mni_unsmoothed.nii.gz',
    'epi_MNI_fullspectrum': '{subject_id}/resting_state/ants/rest_mni_unsmoothed_fullspectrum.nii.gz',
    'moco_parms_file': '{subject_id}/resting_state/realign/rest_realigned.par',
    'jenkinson_file': '{subject_id}/resting_state/realign/rest_realigned_rel.rms',
    'rest2anat_cost_file': '{subject_id}/resting_state/coregister/rest2anat.dat.mincost',
}





# # # # # # # # # #
# LEARNING data_lookup_dict
# # # # # # # # # #
gordon_path = os.path.join(template_dir, 'parcellations/Gordon_2014_Parcels/Parcels_MNI_111_sorted.nii.gz')
craddock_788_path = os.path.join(template_dir,
                                 'parcellations/craddock_2012/scorr_mean_single_resolution/scorr_mean_parc_n_43_k_788_rois.nii.gz')

data_lookup_dict = {}

metrics = {'alff': 'alff/alff.nii.gz',
           'falff': 'alff/falff.nii.gz',
           'alff_z': 'alff_z/alff_zstd.nii.gz',
           'falff_z': 'alff_z/falff_zstd.nii.gz',
           'reho': 'reho/ReHo.nii.gz',
           'variability_std': 'variability/ts_std.nii.gz',
           'variability_std_z': 'variability_z/ts_std_zstd.nii.gz',
           #
           'alff_gm_wm_z': 'alff_gm_wm_z/falff_zstd.nii.gz',
           'alff_gm_wm_z': 'alff_gm_wm_z/alff_zstd.nii.gz',
           'variability_gm_wm_z': 'variability_gm_wm_z/ts_std_zstd.nii.gz',
           }

masks = ['GM', 'WM', 'GM_WM', 'brain_mask']
resolutions = [3, 4, 8]

for m in metrics.keys():
    if m == 'reho':
        use_diagonal = True
    else:
        use_diagonal = False
    for s in [0, 4, 8, 12]:
        for ma in masks:
            for r in resolutions:
                m_str = '%s_%s_%smm_sm%s' % (m, ma, r, s)
                ma_str = '%s_MNI_%smm' % (ma, r)
                data_lookup_dict[m_str] = {
                    'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}', metrics[m]),
                    'mask_name': ma_str, 'use_diagonal': use_diagonal, 'fwhm': s}

resolutions = [5, 4, 3]
hemis = ['lh', 'rh']
metrics = {'ct': 'thickness', 'csa': 'area'}
smoothing = [0, 5, 10, 20]
for h in hemis:
    for m in metrics.keys():
        for r in resolutions:
            for s in smoothing:
                m_str = '%s_%s_fsav%s_sm%s' % (h, m, r, s)
                surf_str = 'surfs/%s.%s.fsaverage%s.%smm.mgz' % (h, metrics[m], r, s)
                data_lookup_dict[m_str] = {
                    'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}', surf_str)}

data_lookup_dict['aseg'] = {'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}/parcstats/aseg')}
for h in ['lh', 'rh']:
    for m in ['area', 'thickness', 'volume']:
        m_str = 'aparc_%s_%s' % (h, m)
        f_str = 'aparc.%s.a2009s.%s' % (h, m)
        data_lookup_dict[m_str] = {'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}/parcstats', f_str)}

data_lookup_dict['craddock_205_noBP'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                         'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                  'con_mat/matrix/bp_None.None/craddock_205/matrix.pkl')}
data_lookup_dict['craddock_205_BP'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                       'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                'con_mat/matrix/bp_0.01.0.1/craddock_205/matrix.pkl')}
data_lookup_dict['craddock_205_BP_scr05'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                             'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                      'con_mat/matrix_scrubbed_0_5/bp_0.01.0.1/craddock_205/matrix.pkl')}

data_lookup_dict['craddock_788_noBP'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                         'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                  'con_mat/matrix/bp_None.None/craddock_788/matrix.pkl')}
data_lookup_dict['craddock_788_BP'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                       'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                'con_mat/matrix/bp_0.01.0.1/craddock_788/matrix.pkl')}
data_lookup_dict['craddock_788_BP_scr05'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                             'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                      'con_mat/matrix_scrubbed_0_5/bp_0.01.0.1/craddock_788/matrix.pkl')}

data_lookup_dict['gordon_noBP'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                   'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                            'con_mat/matrix/bp_None.None/gordon/matrix.pkl')}
data_lookup_dict['gordon_BP'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                 'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                          'con_mat/matrix/bp_0.01.0.1/gordon/matrix.pkl')}

data_lookup_dict['gordon_BP_scr05'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                       'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                'con_mat/matrix_scrubbed_0_5/bp_0.01.0.1/gordon/matrix.pkl')}

data_lookup_dict['gordon_BP_ds'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                    'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                             'con_mat/matrix/bp_0.01.0.1/gordon/matrix_downsampled.pkl'),
                                    'use_diagonal': True}

# msdl abide
data_lookup_dict['msdl_abide_BP'] = {'matrix_name': 'correlation',
                                     'use_fishers_z': True,
                                     'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                              'con_mat/matrix/bp_0.01.0.1/msdl_abide/matrix.pkl'),
                                     'use_diagonal': True}

# Dosenbach
data_lookup_dict['dosenbach'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                 'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                          'con_mat/matrix/bp_0.01.0.1/dosenbach/matrix.pkl')}

# BASC
data_lookup_dict['basc_197'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                         'con_mat/matrix/bp_0.01.0.1/basc_197/matrix.pkl')}
data_lookup_dict['basc_444'] = {'matrix_name': 'correlation', 'use_fishers_z': True,
                                'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                         'con_mat/matrix/bp_0.01.0.1/basc_444/matrix.pkl')}
# BEHAV
data_lookup_dict['behav_wml_wm_total'] = {'df_col_names': ['WM_gesamt'],
                                          'path_str': '{subject_id}'}  # path_str for compatibilty
data_lookup_dict['behav_wml_wmh_norm'] = {'df_col_names': ['wmh_norm'], 'path_str': '{subject_id}'}
data_lookup_dict['behav_wml_wmh_norm_ln'] = {'df_col_names': ['wmh_norm_ln'], 'path_str': '{subject_id}'}
data_lookup_dict['behav_wml_wmh_ln'] = {'df_col_names': ['wmh_ln'], 'path_str': '{subject_id}'}
data_lookup_dict['behav_wml_fazekas'] = {'df_col_names': ['MRT_BefundFazekas'], 'path_str': '{subject_id}'}

# # # # # # # # # #
# template_lookup_dict
# # # # # # # # # #
# get atlas data
template_lookup_dict = {
    'brain_mask_MNI_3mm_frauke': 'Templates/Frauke_Templates/MNI_resampled_brain_mask.nii'
}

for res in [1, 2, 3, 4, 8]:
    for img in ['GM', 'WM', 'GM_WM', 'brain_mask']:
        k = img + '_MNI_' + str(res) + 'mm'
        v = 'MNI152_T1_' + str(res) + 'mm_' + img + '.nii.gz'
        template_lookup_dict[k] = 'Templates/' + v

template_lookup_dict = {k: os.path.join(template_dir, v) for k, v in template_lookup_dict.items()}

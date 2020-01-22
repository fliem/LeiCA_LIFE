import os
import pandas as pd
import numpy as np
from variables import metrics_root_path, subjects_list, subjects_list_folder, in_data_root_path


# EXTRACT NUMBER OF SPIKES (OUTLIERS)
def get_n_spikes(outliers_file):
    try:
        spikes = np.atleast_1d(np.genfromtxt(outliers_file))
        n_spikes = len(spikes)
    except IOError:
        n_spikes = 0
    return n_spikes


# COLLECT FD
mean_FD_P_file_template = os.path.join(metrics_root_path, 'metrics', '{subject_id}/QC/FD_P_mean')
max_FD_P_file_template = os.path.join(metrics_root_path, 'metrics', '{subject_id}/QC/FD_P_max')
ts_FD_P_file_template = os.path.join(metrics_root_path, 'metrics', '{subject_id}/QC/FD_P_ts')
mean_FD_J_file_template = os.path.join(metrics_root_path, 'metrics', '{subject_id}/QC/FD_J_mean')
max_FD_J_file_template = os.path.join(metrics_root_path, 'metrics', '{subject_id}/QC/FD_J_max')
ts_FD_J_file_template = os.path.join(metrics_root_path, 'metrics', '{subject_id}/QC/FD_J_ts')
ts_FD_P_scrubbed_file_template = os.path.join(metrics_root_path, 'metrics', '{subject_id}/QC/FD_P_ts_scrubbed_0.5')

n_spikes_file_template = os.path.join(in_data_root_path,
                                      '{subject_id}/resting_state/denoise/artefact/art.rest2anat_masked_outliers.txt')

df = pd.DataFrame(
    columns=['mean_FD_P', 'max_FD_P', 'mean_FD_J', 'max_FD_J', 'n_TRs', 'n_spikes', 'mean_FD_P_scrubbed_05',
             'max_FD_P_scrubbed_05', 'n_TRs_scrubbed_05'])
FD_ts = None

for i, subject_id in enumerate(subjects_list):
    ts_data = np.loadtxt(ts_FD_P_file_template.format(subject_id=subject_id))
    ts_data_scrubbed = np.loadtxt(ts_FD_P_scrubbed_file_template.format(subject_id=subject_id))
    n_TRs = ts_data.shape[-1]
    n_TRs_scrubbend = ts_data_scrubbed.shape[-1]

    n_spikes = get_n_spikes(n_spikes_file_template.format(subject_id=subject_id))

    mean_FD_P = np.loadtxt(mean_FD_P_file_template.format(subject_id=subject_id)).tolist()
    max_FD_P = np.loadtxt(max_FD_P_file_template.format(subject_id=subject_id)).tolist()
    mean_FD_J = np.loadtxt(mean_FD_J_file_template.format(subject_id=subject_id)).tolist()
    max_FD_J = np.loadtxt(max_FD_J_file_template.format(subject_id=subject_id)).tolist()
    mean_FD_P_scrubbed_05 = ts_data_scrubbed.mean()
    max_FD_P_scrubbed_05 = ts_data_scrubbed.max()

    df.loc[subject_id] = {'mean_FD_P': mean_FD_P,
                          'max_FD_P': max_FD_P,
                          'mean_FD_J': mean_FD_J,
                          'max_FD_J': max_FD_J,
                          'n_TRs': n_TRs,
                          'n_spikes': n_spikes,
                          'mean_FD_P_scrubbed_05': mean_FD_P_scrubbed_05,
                          'max_FD_P_scrubbed_05': max_FD_P_scrubbed_05,
                          'n_TRs_scrubbed_05': n_TRs_scrubbend}

df.to_pickle(os.path.join(subjects_list_folder, 'LIFE_subjects_QC_n%s.pkl' % len(df)))
df.to_excel(os.path.join(subjects_list_folder, 'LIFE_subjects_QC_n%s.xlsx' % len(df)))

'''

'''
import os, pickle
import pandas as pd
from metrics.downsample import get_mapping_gordon, ds_mat
from variables import template_dir, in_data_root_path, subjects_list, metrics_root_path, wd_root_path, \
    selectfiles_templates
from datetime import datetime
import numpy as np

df_file = os.path.join(template_dir, 'parcellations/Gordon_2014_Parcels/Parcels_sorted.txt')
mapping = get_mapping_gordon(df_file)

df_mapping = pd.DataFrame([], columns=['rs_name'])
df_mapping.index.name = 'ds_rs_number'
for i, m_name in enumerate(mapping.keys()):
    df_mapping.loc[i] = m_name
df_mapping.to_csv(os.path.join(template_dir, 'parcellations/Gordon_2014_Parcels/Parcels_sorted_downsampled.txt'))


bp_list = ['bp_0.01.0.1', 'bp_None.None']

t1 = datetime.now()
for subject_id in subjects_list:
    print(subject_id)
    for bp in bp_list:
        m_file = 'PATH/metrics/{subject_id}/con_mat/matrix/{bp}/gordon/matrix.pkl'.format(
            subject_id=subject_id, bp=bp)
        m = pickle.load(open(m_file))['correlation']

        out_path = 'PATH/metrics/{subject_id}/con_mat/matrix/{bp}/gordon/'.format(
            subject_id=subject_id, bp=bp)
        #os.makedirs(out_path)

        # orig matrix
        suf_str = ''
        m_ds = dict()
        m_ds['correlation'] = ds_mat(m, mapping)
        out_file = os.path.join(out_path, 'matrix_downsampled%s.pkl' % suf_str)
        pickle.dump(m_ds, open(out_file, 'w'))


        # abs matrix
        suf_str = '_abs'
        m_ds = dict()
        m_abs = np.abs(m)
        m_ds['correlation'] = ds_mat(m_abs, mapping)
        out_file = os.path.join(out_path, 'matrix_downsampled%s.pkl' % suf_str)
        pickle.dump(m_ds, open(out_file, 'w'))

        # abs matrix
        suf_str = '_pos'
        m_ds = dict()
        m_pos_thr = m.copy()
        m_pos_thr[m_pos_thr < 0] = 0
        m_ds['correlation'] = ds_mat(m_pos_thr, mapping)
        out_file = os.path.join(out_path, 'matrix_downsampled%s.pkl' % suf_str)
        pickle.dump(m_ds, open(out_file, 'w'))


t2 = datetime.now()
d = t2 - t1
print(d.seconds)

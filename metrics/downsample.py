import pandas as pd
import numpy as np
from collections import OrderedDict
from itertools import product


def r_to_z(r):
    m_r = r.copy()
    m_r[m_r == 1] = 1 - 1e-15
    m_r[m_r == -1] = -1 + 1e-15
    return np.arctanh(m_r)


z_to_r = lambda z: np.tanh(z)


def ds_mat(m, mapping):
    '''
    takes correlation matrix of shape (p,p) and
    mapping = {'rs_a': [0,1,2], 'rs_b': [3,4]}
    and downsamples to shape (len(mapping), len(mapping))
    for calculation fishers r to z is performed
    z to r transformed matrix is returned
    '''
    m_z = r_to_z(m)
    n_ds = len(mapping)
    m_ds = np.zeros((n_ds, n_ds))
    m_ds.fill(np.nan)

    for i, j, in product(range(n_ds), range(n_ds)):
        rs_i, rs_j = mapping.keys()[i], mapping.keys()[j]
        ind_i, ind_j = mapping[rs_i], mapping[rs_j]

        m_ds[i, j] = m_z[np.ix_(ind_i, ind_j)].mean()

    return z_to_r(m_ds)


def get_mapping_gordon(df_file, out_file=None):
    df = pd.read_csv(df_file)
    rs_names = np.unique(df.Community.values)
    mapping = OrderedDict()
    for rs in rs_names:
        mapping[rs] = np.where(df.Community.values == rs)[0]

    if out_file:
        df_ds = pd.DataFrame(mapping.keys(), columns=['rs_name'])
        df_ds.index.name = 'ds_index'
        df_ds.to_csv(out_file)
    return mapping

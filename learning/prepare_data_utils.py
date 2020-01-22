###############################################################################################################
# VECTORIZE AND AGGREGATE DATA
# go from single subject original file to numpy array of shape (n_subjects, n_features)
def vectorize_and_aggregate(in_data_file_list, mask_file, matrix_name, parcellation_path, fwhm, use_diagonal,
                            use_fishers_z, df_file, df_col_names):
    import os, pickle
    import numpy as np
    from LeiCA_LIFE.learning.prepare_data_utils import vectorize_ss

    # get an example of the data:
    #save_template: template file; for behav: col names
    vectorized_data, data_type, masker, save_template = vectorize_ss(in_data_file_list[0], mask_file, matrix_name,
                                                                     parcellation_path, fwhm, use_diagonal,
                                                                     use_fishers_z, df_file,
                                                                     df_col_names)
    vectorized_data = np.zeros((len(in_data_file_list), vectorized_data.shape[1]))
    vectorized_data.fill(np.nan)

    for i, in_data_file_ss in enumerate(in_data_file_list):
        vectorized_data[i, :], _, _, _ = vectorize_ss(in_data_file_ss, mask_file, matrix_name, parcellation_path, fwhm,
                                                      use_diagonal, use_fishers_z, df_file, df_col_names)

    vectorized_aggregated_file = os.path.abspath('vectorized_aggregated_data.npy')
    np.save(vectorized_aggregated_file, vectorized_data)

    unimodal_backprojection_info = {'data_type': data_type,
                                    'masker': masker,
                                    'save_template': save_template
                                    }
    unimodal_backprojection_info_file = os.path.abspath('unimodal_backprojection_info.pkl')
    pickle.dump(unimodal_backprojection_info, open(unimodal_backprojection_info_file, 'w'))
    return vectorized_aggregated_file, unimodal_backprojection_info_file


def vectorize_ss(in_data_file, mask_file, matrix_name, parcellation_path, fwhm, use_diagonal, use_fishers_z, df_file,
                 df_col_names):
    import os
    import numpy as np
    from LeiCA_LIFE.learning.prepare_data_utils import _vectorize_nii, _vectorize_matrix, _vectorize_fs, \
        _vectorize_fs_tab, _vectorize_behav_df

    masker = None
    save_template = in_data_file
    if in_data_file.endswith('.nii.gz'):  # 3d nii files
        vectorized_data, masker = _vectorize_nii(in_data_file, mask_file, parcellation_path, fwhm)
        data_type = '3dnii'

    elif in_data_file.endswith('.pkl') & (df_col_names is None):  # pickled matrix files
        vectorized_data = _vectorize_matrix(in_data_file, matrix_name, use_diagonal)
        data_type = 'matrix'

    elif in_data_file.endswith('.mgz'):  # freesurfer: already vetorized
        vectorized_data = _vectorize_fs(in_data_file)
        data_type = 'fs_cortical'

    elif os.path.basename(in_data_file).startswith('aseg') | os.path.basename(in_data_file).startswith(
            'aparc'):  # aseg: just export values from df
        vectorized_data = _vectorize_fs_tab(in_data_file)
        data_type = 'fs_tab'

    elif df_col_names:  # X from df behav
        # subject is inputted via in_data_file
        vectorized_data = _vectorize_behav_df(df_file=df_file, subject=in_data_file,
                                                             df_col_names=df_col_names)
        data_type = 'behav'
        save_template = df_col_names

    else:
        raise Exception('Cannot guess type from filename: %s' % in_data_file)

    def r_to_z(r):
        r = np.atleast_1d(r)
        r[r == 1] = 1 - 1e-15
        r[r == -1] = -1 + 1e-15
        return np.arctanh(r)

    if use_fishers_z:
        vectorized_data = r_to_z(vectorized_data)

    vectorized_data = np.atleast_2d(vectorized_data)
    return vectorized_data, data_type, masker, save_template


###############################################################################################################
# VECTORIZE DATA helper functions
# to bring data from data_type specific file to numpy array

def _vectorize_nii(in_data_file, mask_file, parcellation_path, fwhm):
    from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
    import nibabel as nib

    if parcellation_path is None:
        masker = NiftiMasker(mask_img=mask_file, smoothing_fwhm=fwhm)
    else:
        masker = NiftiLabelsMasker(labels_img=parcellation_path, smoothing_fwhm=fwhm)

    vectorized_data = masker.fit_transform(in_data_file)
    return vectorized_data, masker


def _vectorize_matrix(in_data_file, matrix_name, use_diagonal=False):
    import numpy as np
    import pickle
    def _lower_tria_vector(m, use_diagonal=False):
        '''
        use_diagonal=False: returns lower triangle of matrix (without diagonale) as vector
        use_diagonal=True: returns lower triangle of matrix (with diagonale) as vector; e.g. for downsampled matrices
        matrix dims are
            x,y,subject for aggregated data
            or  x,y for single subject data
        '''
        i = np.ones_like(m).astype(np.bool)
        if use_diagonal:
            k = 0
        else:
            k = -1
        tril_ind = np.tril(i, k)

        if m.ndim == 3:  # subjects alredy aggregated
            vectorized_data = m[tril_ind].reshape(m.shape[0], -1)
        else:  # single subject matrix
            vectorized_data = m[tril_ind]
        return vectorized_data

    # load pickled matrix
    with open(in_data_file, 'r') as f:
        matrix = pickle.load(f)

    # get lower triangle
    vectorized_data = _lower_tria_vector(matrix[matrix_name], use_diagonal=use_diagonal)
    return vectorized_data


def _vectorize_fs(in_data_file):
    import numpy as np
    import nibabel as nb

    img = nb.load(in_data_file)
    in_data = img.get_data().squeeze()
    vectorized_data = in_data[np.newaxis, ...]

    return vectorized_data


def _vectorize_fs_tab(in_data_file):
    import pandas as pd
    df = pd.read_csv(in_data_file, index_col=0, delimiter='\t')
    vectorized_data = df.values
    return vectorized_data


def _vectorize_behav_df(df_file, subject, df_col_names):
    import pandas as pd
    import os
    df = pd.read_pickle(df_file)
    vectorized_data = df.loc[subject][df_col_names].values
    return vectorized_data


###############################################################################################################
# MATRIX TO VECTOR operations
def vector_to_matrix(v, use_diagonal=False):
    '''
    Takes vector of length vector_size and creates 2D matrix to fill off-diagonal cells
    vector_size = matrix_size*(matrix_size-1)*.5
    matrix diagonal is set to 0
    '''
    import numpy as np
    v = np.squeeze(v)
    vector_size = v.shape[0]
    if use_diagonal:
        diag_add = -1
        k = 0
    else:
        diag_add = 1
        k = -1

    matrix_size = int(0.5 * (np.sqrt(8 * vector_size + 1) + diag_add))

    m = np.zeros((matrix_size, matrix_size))
    i = np.ones_like(m).astype(np.bool)

    tril_ind = np.tril(i, k)
    m[tril_ind] = v
    m_sym = m + m.T

    return m_sym


def matrix_to_vector(m):
    '''
    returns lower triangle of 2D matrix (without diagonale) as vector
    '''
    import numpy as np
    i = np.ones_like(m).astype(np.bool)
    tril_ind = np.tril(i, -1)
    v = m[tril_ind]
    return v


def test_vector_to_matrix():
    '''
    tests vector_to_matrix() and matrix_to_vector()
    '''
    import numpy as np
    # simulate data
    matrix_size = 200
    m_in = np.random.randn(matrix_size, matrix_size)
    m_in_sym = m_in + m_in.T
    np.fill_diagonal(m_in_sym, 0)
    v = matrix_to_vector(m_in_sym)
    m_out = vector_to_matrix(v)
    assert np.all(m_out == m_in_sym), "test of vector_to_matrix failed"
    assert m_out[2, 3] == m_out[3, 2], "out matix not symmetrical"

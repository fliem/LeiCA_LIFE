def calc_variability(in_file):
    '''
    input: preprocessed time series
    returns list of filenames with
    out_file
    cf. https://github.com/stefanschmidt/vartbx/blob/master/Variability/shared_var.m
    '''

    import nibabel as nb
    import numpy as np
    import os

    def save_to_nii(out_file_name, out_data, template_img):
        out_img = nb.Nifti1Image(out_data, template_img.get_affine())
        out_img.to_filename(out_file_name)

    img = nb.load(in_file)
    ts = img.get_data()

    out_file_std = os.path.abspath('ts_std.nii.gz')
    ts_std = np.std(ts, 3)
    save_to_nii(out_file_std, ts_std, img)

    return out_file_std


def extract_parcellation_time_series(in_data, parcellation_name, parcellations_dict, bp_freqs, tr):
    '''
    Depending on parcellation['is_probabilistic'] this function chooses either NiftiLabelsMasker or NiftiMapsMasker
    to extract the time series of each parcel
    if bp_freq: data is band passfiltered at (hp, lp), if (None,None): no filter, if (None, .1) only lp...
    tr in ms (e.g. from freesurfer ImageInfo())
    returns np.array with parcellation time series and saves this array also to parcellation_time_series_file, and
    path to pickled masker object
    '''
    from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker, NiftiSpheresMasker
    import os, pickle
    import numpy as np

    if parcellations_dict[parcellation_name]['is_probabilistic'] == True:  # use probab. nilearn
        masker = NiftiMapsMasker(maps_img=parcellations_dict[parcellation_name]['nii_path'], standardize=True)

    elif parcellations_dict[parcellation_name]['is_probabilistic'] == 'sphere':
        atlas = pickle.load(open(parcellations_dict[parcellation_name]['nii_path']))
        coords = atlas.rois
        masker = NiftiSpheresMasker(coords, radius=5, allow_overlap=True, standardize=True)

    else:  # 0/1 labels
        masker = NiftiLabelsMasker(labels_img=parcellations_dict[parcellation_name]['nii_path'],
                                   standardize=True)

    # add bandpass filter (only executes if freq not None
    hp, lp = bp_freqs
    masker.low_pass = lp
    masker.high_pass = hp
    if tr is not None:
        masker.t_r = tr
    else:
        masker.t_r = None

    masker.standardize = True

    masker_file = os.path.join(os.getcwd(), 'masker.pkl')
    with open(masker_file, 'w') as f:
        pickle.dump(masker, f)

    parcellation_time_series = masker.fit_transform(in_data)

    parcellation_time_series_file = os.path.join(os.getcwd(), 'parcellation_time_series.npy')
    np.save(parcellation_time_series_file, parcellation_time_series)

    return parcellation_time_series, parcellation_time_series_file, masker_file


def calculate_connectivity_matrix(in_data, extraction_method):
    '''
    after extract_parcellation_time_series() connectivity matrices are calculated via specified extraction method

    returns np.array with matrixand saves this array also to matrix_file
    '''

    # fixme implement sparse inv covar
    import os, pickle
    import numpy as np

    if extraction_method == 'correlation':
        correlation_matrix = np.corrcoef(in_data.T)
        matrix = {'correlation': correlation_matrix}

    elif extraction_method == 'sparse_inverse_covariance':
        # Compute the sparse inverse covariance
        from sklearn.covariance import GraphLassoCV
        estimator = GraphLassoCV()
        estimator.fit(in_data)
        matrix = {'covariance': estimator.covariance_,
                  'sparse_inverse_covariance': estimator.precision_}

    else:
        raise (Exception('Unknown extraction method: %s' % extraction_method))

    matrix_file = os.path.join(os.getcwd(), 'matrix.pkl')
    with open(matrix_file, 'w') as f:
        pickle.dump(matrix, f)

    return matrix, matrix_file


def get_good_trs(fd_file, fd_thresh):
    import os, numpy as np

    fd = np.genfromtxt(fd_file)
    good_trs = (fd <= fd_thresh)
    fd_scrubbed = fd[good_trs]
    fd_scrubbed_file = os.path.abspath(os.path.basename(fd_file) + '_scrubbed_%.1f' % fd_thresh)
    np.savetxt(fd_scrubbed_file, fd_scrubbed)
    return good_trs, fd_scrubbed_file


def parcellation_time_series_scrubbing(parcellation_time_series_file, good_trs):
    import os, numpy as np

    parcellation_time_series = np.load(parcellation_time_series_file)
    parcellation_time_series_scrubbed = parcellation_time_series[good_trs, ...]
    return parcellation_time_series_scrubbed

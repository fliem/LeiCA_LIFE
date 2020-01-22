###############################################################################################################
# DATA WRANGLING
# aggregate and select data
def aggregate_multimodal_metrics_fct(multimodal_list, unimodal_lookup_dict):
    import numpy as np
    import os, pickle

    multimodal_backprojection_info = []

    X_multimodal = None
    for unimodal_name in multimodal_list:
        this_unimod_backprojection_info = {}

        unimodal_X_file = unimodal_lookup_dict[unimodal_name]['X_file']
        X_unimodal = np.load(unimodal_X_file)
        unimodal_backprojection_info = pickle.load(
            open(unimodal_lookup_dict[unimodal_name]['unimodal_backprojection_info_file']))

        if X_multimodal is None:
            this_unimod_backprojection_info['start_index'] = 0
            X_multimodal = X_unimodal
        else:
            this_unimod_backprojection_info['start_index'] = X_multimodal.shape[1]
            X_multimodal = np.hstack((X_multimodal, X_unimodal))

        this_unimod_backprojection_info['end_index'] = X_multimodal.shape[1]
        this_unimod_backprojection_info['data_type'] = unimodal_backprojection_info['data_type']
        this_unimod_backprojection_info['save_template'] = unimodal_backprojection_info['save_template']
        this_unimod_backprojection_info['masker'] = unimodal_backprojection_info['masker']

        multimodal_backprojection_info.append(this_unimod_backprojection_info)
        this_unimod_backprojection_info['unimodal_name'] = unimodal_name

    multimodal_name = '__'.join(multimodal_list)
    X_multimodal_file = os.path.abspath('X_multimodal.npy')
    np.save(X_multimodal_file, X_multimodal)

    return X_multimodal_file, multimodal_backprojection_info, multimodal_name


def select_subjects_fct(df_all_subjects_pickle_file, subjects_selection_crit_dict, selection_criterium):
    import pandas as pd
    import os
    import numpy as np

    df = pd.read_pickle(df_all_subjects_pickle_file)

    #   EXCLUSION HERE:
    q_str = '(' + ') & ('.join(subjects_selection_crit_dict[selection_criterium]) + ')'
    df['select'] = False
    selected_subjects_list = df.query(q_str).index.values
    df.loc[selected_subjects_list, ['select']] = True
    subjects_selection_index = df['select'].values

    df_use = df.loc[subjects_selection_index]
    df_use_file = os.path.abspath('df_use.csv')
    df_use.to_csv(df_use_file)
    df_use_pickle_file = os.path.abspath('df_use.pkl')
    df_use.to_pickle(df_use_pickle_file)

    return df_use_file, df_use_pickle_file, subjects_selection_index


def select_multimodal_X_fct(X_multimodal_file, subjects_selection_index):
    import os, numpy as np
    X_multimodal = np.load(X_multimodal_file)
    X_multimodal_selected = X_multimodal[subjects_selection_index, :]
    X_multimodal_selected_file = os.path.abspath('X_multimodal_selected.npy')
    np.save(X_multimodal_selected_file, X_multimodal_selected)
    return X_multimodal_selected_file


###############################################################################################################
# PREDICTION
# and helpers for residualizing and plotting
def run_prediction_split_fct(X_file, target_name, selection_criterium, df_file, data_str, regress_confounds=False,
                             run_cv=False, n_jobs_cv=1, run_tuning=False, X_file_nki=None, df_file_nki=None,
                             reverse_split=False, random_state_nki=666, run_learning_curve=False, life_test_size=0.5):
    import os, pickle
    import numpy as np
    import pandas as pd
    from sklearn.svm import SVR
    from sklearn.cross_validation import cross_val_predict, train_test_split, StratifiedKFold
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler, Imputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.utils import shuffle
    from LeiCA_LIFE.learning.learning_utils import pred_real_scatter, plot_brain_age, residualize_group_data, \
        learning_curve_plot

    if X_file_nki:
        run_2sample_training = True
    else:
        run_2sample_training = False

    empty_file = os.path.abspath('empty.txt')
    with open(empty_file, 'w') as f:
        f.write('')

    data_str = target_name + '__' + selection_criterium + '__' + data_str

    variables = ['train_mae', 'train_r2', 'cv_r2', 'cv_mae', 'cv_r2_mean', 'cv_r2_std', 'y_predicted_cv']
    for v in variables:
        try:
            exec (v)
        except NameError:
            exec ('%s = np.nan' % v)


    #########
    # LIFE
    #########
    #### LOAD DATA
    df_life = pd.read_pickle(df_file)
    # add ouput cols to df
    df_life['study'] = 'life'
    df_life['split_group'] = ''

    X_life = np.load(X_file)
    y_life = df_life[[target_name]].values.squeeze()
    confounds_life = df_life[['mean_FD_P']].values
    ind_life = range(X_life.shape[0])

    #### split with age stratification
    n_age_bins = 20
    df_life['age_bins'] = pd.cut(df_life['age'], n_age_bins, labels=range(n_age_bins))

    X_train_life, X_test_life, y_train_life, y_test_life, confounds_train_life, confounds_test_life, \
    ind_train_life, ind_test_life = train_test_split(X_life,
                                                     y_life,
                                                     confounds_life,
                                                     ind_life,
                                                     stratify=df_life['age_bins'].values,
                                                     test_size=life_test_size,
                                                     random_state=666)
    if reverse_split:
        X_train_life, X_test_life = X_test_life, X_train_life
        y_train_life, y_test_life = y_test_life, y_train_life
        confounds_train_life, confounds_test_life = confounds_test_life, confounds_train_life
        ind_train_life, ind_test_life = ind_test_life, ind_train_life

    df_life.ix[ind_train_life, 'split_group'] = 'train'
    df_life.ix[ind_test_life, 'split_group'] = 'test'
    df_train_life = df_life.ix[ind_train_life, ['age_bins', 'study']]
    df_test_life = df_life.ix[ind_test_life, ['age_bins', 'study']]
    sample_weights_train_life = np.ones_like(y_train_life)

    #########
    # NKI
    #########
    if run_2sample_training:
        #### LOAD DATA
        df_nki = pd.read_pickle(df_file_nki)
        df_nki['study'] = 'nki'
        df_nki['train_group_2samp'] = False

        X_nki = np.load(X_file_nki)
        y_nki = df_nki[[target_name]].values.squeeze()
        confounds_nki = df_nki[['mean_FD_P']].values
        ind_nki = range(X_nki.shape[0])

        #### split with age stratification
        df_nki['age_bins'] = pd.cut(df_nki['age'], n_age_bins, labels=range(n_age_bins))

        X_train_nki, X_test_nki, y_train_nki, y_test_nki, confounds_train_nki, confounds_test_nki, \
        ind_train_nki, ind_test_nki = train_test_split(X_nki,
                                                       y_nki,
                                                       confounds_nki,
                                                       ind_nki,
                                                       stratify=df_nki['age_bins'].values,
                                                       train_size=0.1,
                                                       random_state=random_state_nki)
        if reverse_split:
            X_train_nki, X_test_nki = X_test_nki, X_train_nki
            y_train_nki, y_test_nki = y_test_nki, y_train_nki
            confounds_train_nki, confounds_test_nki = confounds_test_nki, confounds_train_nki
            ind_train_nki, ind_test_nki = ind_test_nki, ind_train_nki

        df_nki['train_group_2samp'] = np.nan
        df_nki.ix[ind_train_nki, 'train_group_2samp'] = True
        df_train_nki = df_nki.ix[ind_train_nki, ['age_bins', 'study']]


    else:
        X_train_nki = np.array([])
        y_train_nki = np.array([])
        confounds_train_nki = []
        df_nki = pd.DataFrame([])
        df_train_nki = pd.DataFrame([])
        df_life['train_group_2samp'] = np.nan

    #########
    # stack life and nki
    #########
    df_big = pd.concat((df_life, df_nki))
    df_big_ind_train = ((df_big.split_group == 'train') | (df_big.train_group_2samp == True))
    df_big_ind_test_life = ((df_big.split_group == 'test') & (df_big.study == 'life'))
    df_big['run_2sample_training'] = run_2sample_training

    df_train = pd.concat((df_train_life, df_train_nki))
    df_test = df_test_life

    if run_2sample_training:
        # calc sample weights so that Nl*Wl == Nn*Wn (& (Nl*Wl + Nn*Wn) == Nl + Nn)
        Nl = y_train_life.shape[0]
        Nn = y_train_nki.shape[0]
        w_life = float(Nl + Nn) / (2. * Nl)
        w_nki = float(Nl + Nn) / (2. * Nn)

        sample_weights_train_life = np.zeros_like(y_train_life)
        sample_weights_train_life.fill(w_life)
        sample_weights_train_nki = np.zeros_like(y_train_nki)
        sample_weights_train_nki.fill(w_nki)
        sample_weights_train = np.concatenate((sample_weights_train_life, sample_weights_train_nki))

        X_train = np.concatenate((X_train_life, X_train_nki))
        y_train = np.concatenate((y_train_life, y_train_nki))
        confounds_train = np.concatenate((confounds_train_life, confounds_train_nki))
    else:
        X_train = X_train_life
        y_train = y_train_life
        confounds_train = confounds_train_life
        sample_weights_train = np.ones_like(y_train_life)
    df_train['sample_weights'] = sample_weights_train

    stacked_ind_train_life = slice(0, X_train_life.shape[0])
    stacked_ind_train_nki = slice(X_train_life.shape[0], X_train.shape[0])
    age_bins_train = df_big.ix[df_big_ind_train, 'age_bins'].values

    # test with life data only
    X_test = X_test_life
    y_test = y_test_life
    confounds_test = confounds_test_life



    #########
    # PIPELINE
    #########
    #### REGRESS OUT CONFOUNDS IF NEEDED
    if regress_confounds:
        X_train = residualize_group_data(X_train, confounds_train)
        X_test = residualize_group_data(X_test, confounds_test)

    #### PREPROCESSING
    fill_missing = Imputer()
    var_thr = VarianceThreshold()
    normalize = StandardScaler()

    #### set C to values
    if 'aseg' in data_str:
        C = 1
    else:
        C = 10 ** -3

    regression_model = SVR(kernel='linear', C=C, cache_size=1000)
    pipeline_list = [('fill_missing', fill_missing),
                     ('var_thr', var_thr),
                     ('normalize', normalize)]

    pipeline_list.append(('regression_model', regression_model))
    pipe = Pipeline(pipeline_list)




    #### FIT MODEL
    pipe.fit(X=X_train, y=y_train, **{'regression_model__sample_weight': sample_weights_train})  # (X_train, y_train)
    y_predicted_train = pipe.predict(X_train)
    y_predicted = pipe.predict(X_test)

    df_train['pred_age_train'] = y_predicted_train
    df_test['pred_age_test'] = y_predicted

    y_predicted_train_life = y_predicted_train[stacked_ind_train_life]
    y_predicted_train_nki = y_predicted_train[stacked_ind_train_nki]
    df_life.ix[ind_train_life, 'pred_age_train'] = y_predicted_train_life
    df_life.ix[ind_train_life, 'pred_age_train'] = y_predicted_train_life

    # df_life.ix[ind_train, ['pred_age_train']] = y_predicted_train
    # df_life.ix[ind_test, ['pred_age_test']] = y_predicted
    test_mae = mean_absolute_error(y_test, y_predicted)
    test_r2 = r2_score(y_test, y_predicted)
    test_rpear2 = np.corrcoef(y_test, y_predicted)[0, 1] ** 2

    train_mae = mean_absolute_error(y_train, y_predicted_train)
    train_r2 = r2_score(y_train, y_predicted_train)




    #### RUN CROSSVALIDATION
    if run_cv:
        strat_k_fold = StratifiedKFold(df_train.ix[:, 'age_bins'].values, n_folds=5, shuffle=True, random_state=0)
        # crossval predict and manually calc. cv score to get y_cv_predicted
        # cv_score_ = cross_val_score(pipe, X_train, y_train, cv=strat_k_fold, n_jobs=n_jobs_cv)  #
        y_predicted_cv = cross_val_predict(pipe, X_train, y_train, cv=strat_k_fold, n_jobs=n_jobs_cv)
        df_train['y_predicted_cv'] = y_predicted_cv

        cv_r2 = []
        cv_mae = []
        cv_test_fold = np.zeros_like(y_train)
        cv_test_fold.fill(np.nan)
        for k, (k_train, k_test) in enumerate(strat_k_fold):
            cv_r2.append(r2_score(y_train[k_test], y_predicted_cv[k_test]))
            cv_mae.append(mean_absolute_error(y_train[k_test], y_predicted_cv[k_test]))
            cv_test_fold[k_test] = k
        cv_r2_mean = np.mean(cv_r2)
        cv_r2_std = np.std(cv_r2)

        # df_big['cv_test_fold'] = np.nan
        # df_big.ix[df_big_ind_train, 'cv_test_fold'] = cv_test_fold
        df_train['cv_test_fold'] = cv_test_fold

    if run_learning_curve:
        X_full = np.vstack((X_train_life, X_test_life))
        y_full = np.hstack((y_train_life, y_test_life))
        from sklearn.learning_curve import learning_curve
        train_sizes, train_scores, test_scores = learning_curve(pipe, X_full, y_full, cv=5, n_jobs=n_jobs_cv,
                                                                train_sizes=np.linspace(0.1, 1.0, 10))

        learning_curve_plot_file, learning_curve_df_file = learning_curve_plot(train_sizes, train_scores, test_scores,
                                                                               'learning curve', data_str,
                                                                               post_str='_training_curve')
    else:
        learning_curve_plot_file, learning_curve_df_file = empty_file, empty_file


    #### SCATTER PLOTS
    title_str = 'r2: {:.3f} MAE:{:.3f}'.format(test_r2, test_mae)
    scatter_file = pred_real_scatter(y_test, y_predicted, title_str, data_str)

    if run_cv:
        title_str = 'r2: {:.3f}({:.3f}) MAE:{:.3f}({:.3f})'.format(cv_r2_mean, cv_r2_std, np.mean(cv_mae),
                                                                   np.std(cv_mae))
        scatter_file_cv = pred_real_scatter(y_train, y_predicted_cv, title_str, data_str, post_str='_cv')

    else:
        scatter_file_cv = empty_file

    brain_age_scatter_file = plot_brain_age(y_test, y_predicted, data_str)






    #### TUNING CURVES
    if run_tuning:
        from sklearn.learning_curve import validation_curve
        from sklearn.cross_validation import StratifiedKFold
        import pylab as plt
        strat_k_fold = StratifiedKFold(df_big.ix[df_big_ind_train, 'age_bins'].values, n_folds=5, shuffle=True,
                                       random_state=0)
        param_range = np.logspace(-4, 0, num=12)
        # fixme n_jobs
        train_scores, test_scores = validation_curve(pipe, X_train, y_train, param_name="regression_model__C",
                                                     param_range=param_range,
                                                     cv=strat_k_fold, n_jobs=n_jobs_cv)
        # plot
        # http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#example-model-selection-plot-validation-curve-py
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.figure()
        plt.title("Validation Curve")
        plt.xlabel("C")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.2,
                         color="r")
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="g")
        plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                         color="g")
        plt.legend(loc="best")

        tuning_curve_file = os.path.join(os.getcwd(), 'tuning_curve_' + data_str + '.pdf')
        plt.savefig(tuning_curve_file)
        plt.close()
        # #############################################
    else:
        tuning_curve_file = empty_file





    #### join df_train and df_test with df_big
    # drop duplicate columns
    df_train.drop([l for l in df_train.columns if l in df_big.columns], axis=1, inplace=True)
    df_test.drop([l for l in df_test.columns if l in df_big.columns], axis=1, inplace=True)
    df_big = df_big.join(df_train, how='left')
    df_big = df_big.join(df_test, how='left')

    #### SAVE
    df_res_out_file = os.path.abspath(data_str + '_df_results.pkl')
    df_res = pd.DataFrame(
        {'FD_res': regress_confounds, 'r2_train': [train_r2], 'MAE_train': [train_mae], 'r2_test': [test_r2],
         'rpear2_test': [test_rpear2], 'MAE_test': [test_mae], 'cv_r2': [cv_r2], 'cv_r2_mean': [cv_r2_mean],
         'cv_r2_std': [cv_r2_std]},
        index=[data_str])
    df_res.to_pickle(df_res_out_file)

    df_life_out_file = os.path.join(os.getcwd(), data_str + '_life_df_predicted.pkl')
    df_life.to_pickle(df_life_out_file)

    df_nki_out_file = os.path.join(os.getcwd(), data_str + '_nki_df_predicted.pkl')
    df_nki.to_pickle(df_nki_out_file)

    df_big_out_file = os.path.join(os.getcwd(), data_str + '_df_predicted.pkl')
    df_big.to_pickle(df_big_out_file)

    model_out_file = os.path.join(os.getcwd(), 'trained_model.pkl')
    with open(model_out_file, 'w') as f:
        pickle.dump(pipe, f)

    return scatter_file, brain_age_scatter_file, df_life_out_file, df_nki_out_file, df_big_out_file, model_out_file, df_res_out_file, \
           tuning_curve_file, scatter_file_cv, learning_curve_plot_file, learning_curve_df_file


def residualize_group_data(signals, confounds):
    '''
    regresses out confounds from signals
    signals.shape: subjects x n_data_points
    confounds.shape: subjects x n_confounds
    returns residualized_signals.shape: subjects x n_data_points
    '''
    from nilearn.signal import clean
    import numpy as np
    counfounds_plus_constant = np.concatenate((confounds, np.ones((confounds.shape[0], 1))), axis=1)
    residualized_signals = clean(signals, detrend=False, standardize=False, confounds=counfounds_plus_constant,
                                 low_pass=None, high_pass=None, t_r=None)
    return residualized_signals


def pred_real_scatter(y_test, y_test_predicted, title_str, in_data_name, post_str=''):
    import os
    import pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.figure()
    plt.scatter(y_test, y_test_predicted)
    plt.plot([10, 80], [10, 80], 'k')
    plt.xlabel('real')
    plt.ylabel('predicted')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title(title_str)
    plt.tight_layout()
    scatter_file = os.path.join(os.getcwd(), 'scatter_' + in_data_name + post_str + '.pdf')
    pp = PdfPages(scatter_file)
    pp.savefig()
    pp.close()
    return scatter_file


def plot_brain_age(y_test, y_test_predicted, in_data_name):
    import os
    import pylab as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    brain_age = y_test_predicted - y_test
    title_str = 'mean brain age: %s\nr = %s' % (brain_age.mean(), np.corrcoef(brain_age, y_test)[0, 1])
    plt.figure()
    plt.scatter(y_test, brain_age)
    plt.xlabel('real')
    plt.ylabel('brain_age (pred-real)')
    plt.title(title_str)
    plt.tight_layout()
    scatter_file = os.path.join(os.getcwd(), 'scatter_brain_age_' + in_data_name + '.pdf')
    pp = PdfPages(scatter_file)
    pp.savefig()
    pp.close()
    return scatter_file


def learning_curve_plot(train_sizes, train_scores, test_scores, title_str, in_data_name, post_str=''):
    import os, numpy as np
    import pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")

    plot_file = os.path.join(os.getcwd(), 'learning_curve_' + in_data_name + post_str + '.pdf')
    df_file = os.path.join(os.getcwd(), 'learning_curve_' + in_data_name + post_str + '.txt')
    pp = PdfPages(plot_file)
    pp.savefig()
    pp.close()
    df_learning_curve = pd.DataFrame({'train_sizes': train_sizes, 'train_scores_mean':
        train_scores_mean, 'train_scores_std': train_scores_std, 'test_scores_mean': test_scores_mean,
                                      'test_scores_std': test_scores_std})
    df_learning_curve.to_csv(df_file)

    return plot_file, df_file


###############################################################################################################
# BACKPROJECTION
def backproject_and_split_weights_fct(trained_model_file, multimodal_backprojection_info, data_str, target_name):
    from LeiCA_LIFE.learning.learning_utils import backproject_weights_to_full_space, save_weights

    data_str = target_name + '__' + data_str

    out_file_list = []
    out_file_render_list = []

    weights = backproject_weights_to_full_space(trained_model_file)

    for unimodal_backprojection_info in multimodal_backprojection_info:
        out_data = weights[:, unimodal_backprojection_info['start_index']:unimodal_backprojection_info['end_index']]
        data_type = unimodal_backprojection_info['data_type']
        save_template = unimodal_backprojection_info['save_template']
        masker = unimodal_backprojection_info['masker']

        out_name = data_str + '__' + unimodal_backprojection_info['unimodal_name']

        out_file, out_file_render = save_weights(out_data, data_type, save_template, out_name, masker)
        out_file_list.append(out_file)
        out_file_render_list.append(out_file_render)

    return out_file_list, out_file_render_list


def backproject_weights(trained_model):
    '''
    Takes a vector of weights and fills NaNs for all the features that have been eliminated during preprocessing
    '''
    import numpy as np
    import sklearn

    ADD_CONSTANT = 1000

    if isinstance(trained_model, sklearn.grid_search.GridSearchCV):  # necessary because gs object is set up diffrently
        steps = trained_model.best_estimator_.named_steps
    else:
        steps = trained_model.named_steps

    imp = steps['fill_missing']
    var_thr = steps['var_thr']
    scaler = steps['normalize']
    regression_model = steps['regression_model']

    if 'anova_filter' in steps.keys():
        # back trainsform weights from sparse (anova selection) to full
        # insert nans in eliminated features
        anova_filter = steps['anova_filter']
        weights = anova_filter.inverse_transform(regression_model.coef_ + ADD_CONSTANT)
        sparse_ind = (weights == 0)
        weights -= ADD_CONSTANT

    elif 'regression_model' in steps.keys():
        weights = regression_model.coef_
        sparse_ind = np.zeros_like(weights).astype(np.bool)

    else:
        raise Exception('neither rfe nor regression_model found in steps.keys() %s' % steps.keys())

    # add constant to the weights in order to differentiate between 0 weights and zeros that have been iputed in the
    # case of 0-variance features
    # after replacing 0s with nans, substract constant
    mapped_back_var_thr = var_thr.inverse_transform(weights + ADD_CONSTANT)
    mapped_back_var_thr[mapped_back_var_thr == 0] = np.nan
    mapped_back_var_thr -= ADD_CONSTANT

    mapped_back = np.zeros((1, imp.statistics_.shape[0]))
    mapped_back.fill(np.nan)
    mapped_back[0, ~np.isnan(imp.statistics_)] = mapped_back_var_thr

    # set sparse weights to nan
    mapped_back[sparse_ind] = np.nan
    return mapped_back


def backproject_weights_to_full_space(trained_model_file):
    '''
    Loads trained model and call backproject_weights
    '''
    import pickle
    from LeiCA_LIFE.learning.learning_utils import backproject_weights
    trained_model = pickle.load(open(trained_model_file))
    return backproject_weights(trained_model)


def save_weights(data, data_type, save_template, outfile_name, masker=None):
    '''
    saves weights as nii or other file
    '''
    import nibabel as nb
    import numpy as np
    import pandas as pd
    import os
    from nilearn import plotting
    import pylab as plt
    from LeiCA_LIFE.learning.prepare_data_utils import vector_to_matrix

    weights_file_str = '_weights'
    render_file_str = '_rendering'
    outfile_render = os.path.abspath(outfile_name + render_file_str + '.pdf')

    plt.figure()

    if data_type == '3dnii':
        nii = masker.inverse_transform(data)
        outfile = os.path.abspath(outfile_name + weights_file_str + '.nii.gz')
        nii.to_filename(outfile)

        plotting.plot_stat_map(outfile)


    elif data_type == 'matrix':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.npy')
        data = vector_to_matrix(data, use_diagonal=False)
        np.save(outfile, data)

        plt.imshow(data, interpolation='nearest')
        plt.colorbar()

    elif data_type == 'fs_cortical':
        fs_img = nb.load(save_template)
        # Sabina says that's how nb handels this case
        fs_data = fs_img.get_data()
        fs_data[:] = data.reshape(fs_data.shape[0], 1, 1)
        # check whether writing data into image worked
        np.testing.assert_array_almost_equal(fs_img.get_data().squeeze(), data.squeeze())

        outfile = os.path.abspath(outfile_name + weights_file_str + '.mgz')
        fs_img.to_filename(outfile)

        # # fixme DOES NOT WORK
        # from surfer import Brain
        # # replace nans by 0, otherwise pysurfer cannot display them
        # data_nonans = data.copy()
        # data_nonans[np.isnan(data_nonans)] = 0
        # hemi =  outfile_name[outfile_name.find('__')+2:][:2]
        # brain = Brain('fsaverage5', hemi, 'inflated', config_opts={"background": "white"}) #, background="white") ,
        # brain.add_data(data_nonans.squeeze())
        # brain.save_montage(outfile_render, order=['lat', 'med'], orientation='h', border_size=10)
        # brain.close()

    elif data_type == 'fs_tab':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.csv')
        df = pd.read_csv(save_template, index_col=0, delimiter='\t')
        df.index.name = 'subject_id'
        df.drop(df.index[0], axis=0, inplace=True)
        df.loc[outfile_name, :] = data
        df.to_csv(outfile)
        df.T.plot(kind='barh')

    elif data_type == 'behav':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.csv')
        # for behav save_template contains colnames
        df = pd.DataFrame([], columns=save_template)
        df.loc[outfile_name, :] = data
        df.to_csv(outfile)
        df.T.plot(kind='barh')

    else:
        raise Exception('wrong data type %s' % data_type)

    plt.savefig(outfile_render)
    plt.close()

    return outfile, outfile_render


###############################################################################################################
# PREDICTION from trained model
def run_prediction_from_trained_model_fct(trained_model_file, X_file, target_name, selection_criterium, df_file,
                                          data_str, regress_confounds=False):
    import os, pickle
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error
    from LeiCA_LIFE.learning.learning_utils import pred_real_scatter, plot_brain_age, residualize_group_data

    data_str = target_name + '__' + selection_criterium + '__' + data_str

    df = pd.read_pickle(df_file)
    df['pred_age_test'] = np.nan

    X_test = np.load(X_file)
    y_test = df[[target_name]].values.squeeze()
    confounds = df[['mean_FD_P']].values

    # REGRESS OUT CONFOUNDS IF NEEDED
    if regress_confounds:
        X_test = residualize_group_data(X_test, confounds)

    with open(trained_model_file, 'r') as f:
        pipe = pickle.load(f)

    # RUN PREDICTION
    y_predicted = pipe.predict(X_test)

    df.ix[:, ['pred_age_test']] = y_predicted

    test_mae = mean_absolute_error(y_test, y_predicted)
    test_r2 = r2_score(y_test, y_predicted)
    test_rpear2 = np.corrcoef(y_test, y_predicted)[0, 1] ** 2

    train_r2 = np.nan
    train_mae = np.nan


    # SCATTER PLOTS
    title_str = 'r2: {:.3f} MAE:{:.3f}'.format(test_r2, test_mae)
    scatter_file = pred_real_scatter(y_test, y_predicted, title_str, data_str)

    brain_age_scatter_file = plot_brain_age(y_test, y_predicted, data_str)

    df_use_file = os.path.join(os.getcwd(), data_str + '_df_predicted.pkl')
    df.to_pickle(df_use_file)

    # performace results df
    df_res_out_file = os.path.abspath(data_str + '_df_results.pkl')
    df_res = pd.DataFrame(
        {'FD_res': regress_confounds, 'r2_train': [train_r2], 'MAE_train': [train_mae],
         'r2_test': [test_r2], 'rpear2_test': [test_rpear2],
         'MAE_test': [test_mae]},
        index=[data_str])
    df_res.to_pickle(df_res_out_file)

    return scatter_file, brain_age_scatter_file, df_use_file, df_res_out_file

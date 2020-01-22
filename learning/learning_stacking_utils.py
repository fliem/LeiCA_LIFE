import os
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

__author__ = 'franzliem'


def tune_and_train_rf(X_train, y_train, strat_k_fold=None):
    '''
    Uses oob estimates to find optimal max_depth between None + 0...20
    Refits with best max_depth
    '''
    oob_r2 = []
    cv_list = [None] + range(1, 20)
    for md in cv_list:
        rf = RandomForestRegressor(n_estimators=100, max_depth=md, oob_score=True, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        oob_r2.append(rf.oob_score_)

    best_max_depth = cv_list[np.argmax(oob_r2)]
    print("best max_depth: %s" % best_max_depth)

    # CV
    rf = RandomForestRegressor(n_estimators=100, max_depth=best_max_depth, oob_score=True, random_state=0, n_jobs=-1)

    cv_results = None
    if strat_k_fold:
        y_predicted_cv = cross_val_predict(rf, X_train, y_train, cv=strat_k_fold, n_jobs=-1)
        cv_r2 = []
        cv_mae = []
        for k_train, k_test in strat_k_fold:
            cv_r2.append(r2_score(y_train[k_test], y_predicted_cv[k_test]))
            cv_mae.append(mean_absolute_error(y_train[k_test], y_predicted_cv[k_test]))
        cv_results = {'y_predicted_cv': y_predicted_cv,
                      'cv_r2': cv_r2,
                      'cv_mae': cv_mae,
                      'oob_r2': oob_r2}

    # refit
    rf.fit(X_train, y_train)
    return rf, cv_results


def stacking(out_path, target, selection_crit, source_dict, source_selection_dict, rf=None):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    os.chdir(out_path)

    if rf is None:
        run_fitting = True
    else:
        run_fitting = False
        rf_file_template = rf

    df_in = {}
    for s, f in source_dict.items():
        df_in[s] = pd.read_pickle(f)
        df_in[s]['source'] = s

    for stacking_crit in source_selection_dict.keys():
        file_pref = target + '__' + selection_crit + '__' + stacking_crit + '__'

        scores_test = pd.DataFrame([])
        df_all = pd.DataFrame([])
        for s in source_selection_dict[stacking_crit]:
            df_all = pd.concat((df_all, df_in[s]))

        # get one single source example to get age...
        df_single_source = df_in[s]

        # add columns in the case of test-only data
        if 'split_group' not in df_all:
            df_all['split_group'] = 'test'

        for a in ['select', 'y_predicted_cv','sample_weights',  'train_group_2samp', 'study']: # 'no_motion_grp', 'random_motion_grp',
            if a not in df_all:
                df_all[a] = np.nan

        df = df_all[['source', 'age', 'split_group', 'select', 'y_predicted_cv', 'pred_age_test', 'sample_weights', 'train_group_2samp', 'study']] #'no_motion_grp', 'random_motion_grp',


        test_ind = df['split_group'] == 'test'
        df_test = df[test_ind].copy()

        if run_fitting:  # fit rf
            print ('Fitting stacking model')

            train_ind = ((df['split_group'] == 'train') | (df['train_group_2samp'] == True))
            df_train = df[train_ind].copy()

            dd_train = df_train.pivot_table(values='y_predicted_cv', columns='source', index=df_train.index)
            single_sources = dd_train.columns.values
            dd_train['mean_pred'] = dd_train.mean(1)
            dd_train = dd_train.join(df_single_source[['age']], how='left')

            n_age_bins = 20
            dd_train['age_bins_rf'] = pd.cut(dd_train['age'], n_age_bins, labels=range(n_age_bins))
            strat_k_fold = StratifiedKFold(dd_train['age_bins_rf'].values, n_folds=5, shuffle=True, random_state=0)

            X_train, y_train = dd_train[single_sources], dd_train['age']

            # TRAINING
            rf, cv_results = tune_and_train_rf(X_train, y_train, strat_k_fold)

            dd_train['cv_test_fold'] = np.nan
            for k, (k_train, k_test) in enumerate(strat_k_fold):
                dd_train.ix[k_test, 'cv_test_fold'] = k

            fi = pd.DataFrame(rf.feature_importances_, columns=['feature_importances'], index=single_sources)
            plt.figure()
            sns.barplot(fi.feature_importances, fi.index.values)
            plt.xlim([0, 1])
            plt.savefig(os.path.abspath('fi_' + file_pref + '.pdf'))
            plt.close()

            plt.plot(cv_results['oob_r2'])
            plt.vlines(np.argmax(cv_results['oob_r2']), *plt.ylim())
            plt.savefig(os.path.abspath(file_pref + 'oob_max_depth.pdf'))
            plt.close()

            # get train error
            y_predicted_train = rf.predict(X_train)
            r2_train = r2_score(y_train, y_predicted_train)
            mae_train = mean_absolute_error(y_train, y_predicted_train)

            scores_train = pd.DataFrame({'cv_r2': [cv_results['cv_r2']],
                                         'cv_mae': [cv_results['cv_mae']],
                                         'r2_train': r2_train,
                                         'MAE_train': mae_train
                                         }, index=[stacking_crit])

            dd_train['y_predicted_cv'] = cv_results['y_predicted_cv']
            dd_train['pred_age_train'] = y_predicted_train

            plt.figure()
            # plt.scatter(y_train, y_predicted_train)
            f = sns.jointplot('age', 'pred_age_train', data=dd_train, xlim=(10, 90), ylim=(10, 90))
            ax = sns.plt.gca()
            plt.savefig(os.path.abspath('scatter_train_' + file_pref + '_' + '_' + stacking_crit + '_.pdf'))
            plt.close()
        else:
            rf_file = rf_file_template.format(stacking_crit=stacking_crit)
            rf = pickle.load(open(rf_file))
            dd_train = pd.DataFrame([])


        # PREDICTIONS ON TEST SET

        dd_test = df_test.pivot_table(values='pred_age_test', columns='source', index=df_test.index)
        single_sources = dd_test.columns.values

        # add motion groups to df
        # motion_grps = df[['no_motion_grp', 'random_motion_grp']].copy()
        # motion_grps['ind'] = motion_grps.index
        # motion_grps = motion_grps.drop_duplicates()[['no_motion_grp', 'random_motion_grp']]
        # dd_test = dd_test.join(motion_grps, how='left')

        dd_test['mean_pred'] = dd_test.mean(1)
        dd_test = dd_test.join(df_single_source[['age']], how='left')

        X_test, y_test = dd_test[single_sources], dd_test['age']
        dd_test['pred_age_test'] = rf.predict(X_test)

        for m in source_selection_dict[stacking_crit] + ['mean_pred', 'pred_age_test']:
            scores_test.ix[m, 'r2'] = r2_score(dd_test['age'], dd_test[m])
            scores_test.ix[m, 'rpear'] = np.corrcoef(dd_test['age'], dd_test[m])[0, 1]
            scores_test.ix[m, 'rpear2'] = np.corrcoef(dd_test['age'], dd_test[m])[0, 1] ** 2
            scores_test.ix[m, 'mae'] = mean_absolute_error(dd_test['age'], dd_test[m])
            scores_test.ix[m, 'medae'] = median_absolute_error(dd_test['age'], dd_test[m])

            plt.figure()
            plt.scatter(dd_test['age'], dd_test[m])
            # plt.axis('equal')
            plt.plot([10, 90], [10, 90])
            plt.xlim([10, 90]);
            plt.ylim([10, 90])
            plt.title('predictions TEST: %s (%s)\n%.3f' % (m, stacking_crit, scores_test.ix[m, 'r2']))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.abspath('scatter_test_' + file_pref + '_' + '_' + m + '_.pdf'))
            plt.close()

        # # MOTION BALANCING
        # if run_fitting:  # only run if not from a 'predict from trained model' single source stream
        #     X_test_no_motion, y_test_no_motion = dd_test.ix[dd_test.no_motion_grp, single_sources], dd_test.ix[
        #         dd_test.no_motion_grp, 'age']
        #     y_predicted_train_no_motion = rf.predict(X_test_no_motion)
        #     dd_test.ix[dd_test.no_motion_grp, 'pred_age_no_motion'] = y_predicted_train_no_motion
        #
        #     X_test_random_motion, y_test_random_motion = dd_test.ix[dd_test.random_motion_grp, single_sources], \
        #                                                  dd_test.ix[
        #                                                      dd_test.random_motion_grp, 'age']
        #     y_predicted_train_random_motion = rf.predict(X_test_random_motion)
        #     dd_test.ix[dd_test.random_motion_grp, 'pred_age_random_motion'] = y_predicted_train_random_motion
        #
        #     m = 'pred_age_no_motion'
        #     y_true = y_test_no_motion
        #     y_pred = y_predicted_train_no_motion
        #     scores_test.ix[m, 'r2'] = r2_score(y_true, y_pred)
        #     scores_test.ix[m, 'rpear'] = np.corrcoef(y_true, y_pred)[0, 1]
        #     scores_test.ix[m, 'rpear2'] = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        #     scores_test.ix[m, 'mae'] = mean_absolute_error(y_true, y_pred)
        #     scores_test.ix[m, 'medae'] = median_absolute_error(y_true, y_pred)
        #
        #     m = 'pred_age_random_motion'
        #     y_true = y_test_random_motion
        #     y_pred = y_predicted_train_random_motion
        #     scores_test.ix[m, 'r2'] = r2_score(y_true, y_pred)
        #     scores_test.ix[m, 'rpear'] = np.corrcoef(y_true, y_pred)[0, 1]
        #     scores_test.ix[m, 'rpear2'] = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        #     scores_test.ix[m, 'mae'] = mean_absolute_error(y_true, y_pred)
        #     scores_test.ix[m, 'medae'] = median_absolute_error(y_true, y_pred)

        # save
        rf_file = os.path.abspath(file_pref + 'stacking_fitted_model.pkl')
        predictions_file = os.path.abspath(file_pref + 'stacking_df_predicted.pkl')
        scores_train_file = os.path.abspath(file_pref + 'stacking_train_df_results.pkl')
        scores_test_file = os.path.abspath(file_pref + 'stacking_test_df_results.pkl')

        with open(rf_file, 'w') as f:
            if run_fitting:
                pickle.dump(rf, f)
            else:  # if model was not fittet here save empty structure
                pickle.dump(np.nan, f)

        # dd_test.to_pickle(predictions_test_file)
        dd_test['split_group'] = 'test'
        dd_train['split_group'] = 'train'
        dd_big = pd.concat((dd_test, dd_train))
        dd_big.to_pickle(predictions_file)

        scores_test.to_pickle(scores_test_file)

        if run_fitting:
            scores_train.to_pickle(scores_train_file)

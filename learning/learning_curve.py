# from http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html


def learning_curve_fct(X_file, y_file, out_name, model_file):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.learning_curve import learning_curve
    import os, pickle
    from sklearn.svm import SVR
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import Imputer


    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
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
        return plt

    X = np.load(X_file)
    y = np.load(y_file)
    svr = pickle.load(open(model_file))

    title = 'Learning Curves: ' + out_name
    plot_learning_curve(svr, title, X, y, cv=5, n_jobs=5, train_sizes=np.linspace(.5, 1.0, 5))
    curve_file = os.path.join(os.getcwd(), out_name+'_curve.pdf')
    plt.savefig(curve_file)

    return curve_file


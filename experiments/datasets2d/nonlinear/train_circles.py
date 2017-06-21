import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

from multiprocessing import Pool

from pulearn import (
    MultilayerPerceptron,
    WeightedUnlabelledMultilayerPerceptron,
    UnlabelledExponentialLossMultilayerPerceptron,
    HardBootstrappingMultilayerPerceptron
)

from pulearn.utils import synthesize_pu_labels


def fit_classifier(args):
    i = args['iter']
    pct_missing = args['pct_missing']
    wu = args['weight_u']

    X_train = args["X_train"]
    Y_train = args["Y_train"]
    X_test = args["X_test"]
    y_test = args["y_test"]

    clsf = args["clsf"]
    if clsf == 'mlp':
        classifier = MultilayerPerceptron(n_input=n_input,
                          n_classes=n_classes,
                          n_hiddens=[8, 8],
                          learning_rate=5e-3,
                          alpha=0,
                          epochs=100,
                          class_weight=None,
                          verbose=False)
    elif clsf == 'bw_mlp':
        classifier = MultilayerPerceptron(n_input=n_input,
                          n_classes=n_classes,
                          n_hiddens=[8, 8],
                          learning_rate=5e-3,
                          alpha=0,
                          epochs=100,
                          class_weight='balanced',
                          verbose=False)
    elif clsf == 'wu_mlp':
        classifier = WeightedUnlabelledMultilayerPerceptron(n_input=n_input,
                          n_classes=n_classes,
                          n_hiddens=[8, 8],
                          learning_rate=5e-3,
                          alpha=0,
                          epochs=100,
                          class_weight='balanced',
                          unlabelled_weight={0:wu, 1:1},
                          verbose=False)
    elif clsf == 'eu_mlp':
        classifier = UnlabelledExponentialLossMultilayerPerceptron(n_input=n_input,
                          n_classes=n_classes,
                          n_hiddens=[8, 8],
                          learning_rate=5e-3,
                          alpha=0,
                          epochs=200,
                          class_weight=None,
                          unlabelled_weight=None,
                          verbose=False)
    elif clsf == 'hb_mlp':
        classifier = HardBootstrappingMultilayerPerceptron(n_input=n_input,
                          n_classes=n_classes,
                          n_hiddens=[8, 8],
                          learning_rate=1e-2,
                          alpha=0,
                          epochs=200,
                          class_weight="balanced",
                          unlabelled_weight={0:wu, 1:1},
                          verbose=False)

    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    classifier.close_session()
    auc = metrics.roc_auc_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("Iteration", i, "Fitting:", clsf, "Missing:", pct_missing,
        "Unlabelled weight:", wu,
        "Got", "AUC:", auc, "F1-score:", f1, "Accuracy:", acc)
    record = {}
    record['%missing'] = pct_missing
    record['weight_u'] = wu
    record["classifier"] = clsf
    record['AUC'] = auc
    record['F1-score'] = f1
    record['Accuracy'] = acc
    return record


if __name__ == "__main__":

    n_input = 2
    n_classes = 2

    n_iters = 50

    for i in range(n_iters):
        print("Starting {}-th iteration".format(i))

        # Construct dataset
        args_list = []
        X, y = datasets.make_circles(n_samples=800, noise=0.2, factor=.3)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.5)
        Y_train = np.eye(n_classes)[y_train]
        Y_train_pu = synthesize_pu_labels(Y_train, random_state=42, verbose=False)

        # Initial arguments
        for pct_missing in Y_train_pu.keys():
            for wu in [0.5]:
                for clsf in ['mlp', 'bw_mlp', 'wu_mlp', 'hb_mlp', 'eu_mlp']:
                    data = {}
                    data["iter"] = i
                    data["X_train"] = X_train
                    data["Y_train"] = Y_train_pu[pct_missing]
                    data["X_test"] = X_test
                    data["y_test"] = y_test
                    data["pct_missing"] = pct_missing
                    data["clsf"] = clsf
                    data["weight_u"] = wu
                    args_list.append(data)

        # Start jobs
        pool = Pool(processes=8)
        acc_grid_list = pool.map(fit_classifier, args_list)
        acc_grid = pd.DataFrame(acc_grid_list)
        acc_grid.to_csv("pulearn_circles.csv", index=False, mode='a')

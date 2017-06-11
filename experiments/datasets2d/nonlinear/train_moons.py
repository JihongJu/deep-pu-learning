import pandas as pd
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from utils import get_PU_labels

from multiprocessing import Pool

from multilayer_perceptron import MultilayerPerceptron
from pu_learning import (
    ClassDepLossMultilayerPerceptron,
    HardBoostrappingMultilayerPerceptron
)


def fit_classifier(args):
    i = args['iter']
    pct_missing = args['pct_missing']
    w_unlabelled = args['w_unlabelled']

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
              regularization=0,
              training_epochs=200,
              class_weight=[w_unlabelled, 1],
              imbalanced=True,
              verbose=False)
    elif clsf == 'hardb_mlp':
        classifier = HardBoostrappingMultilayerPerceptron(n_input=n_input,
              n_classes=n_classes,
              n_hiddens=[8, 8],
              learning_rate=5e-3,
              regularization=0,
              training_epochs=200,
              class_weight=None,
              betas=[w_unlabelled, 1],
              imbalanced=True,
              verbose=False)
    elif clsf == 'clsdep_mlp':
        classifier = ClassDepLossMultilayerPerceptron(n_input=n_input,
              n_classes=n_classes,
              n_hiddens=[8, 8],
              learning_rate=5e-3,
              regularization=0,
              training_epochs=200,
              class_weight=[w_unlabelled, 1],
              imbalanced=True,
              verbose=False)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    classifier.close_session()
    auc = metrics.roc_auc_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    acc = metrics.accuracy_score(y_test, y_pred)
    print("Iteration", i, "Fitting:", clsf, "Missing:", pct_missing,
        "Unlabelled weight:", w_unlabelled,
        "Got", "AUC:", auc, "F1-score:", f1, "Accuracy:", acc)
    record = {}
    record['%missing'] = pct_missing
    record['w_unlabelled'] = w_unlabelled
    record["classifier"] = clsf
    record['AUC'] = auc
    record['F1-score'] = f1
    record['Accuracy'] = acc
    return record


if __name__ == "__main__":

    n_input = 2
    n_classes = 2

    n_iters = 10

    for i in range(n_iters):
        print("Starting {}-th iteration".format(i))

        # Construct dataset
        args_list = []
        X, y = datasets.make_moons(n_samples=800, noise=0.2)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.5)
        Y_train = np.eye(n_classes)[y_train]
        Y_train_pu = get_PU_labels(Y_train, random_seed=42, verbose=False)

        # Initial arguments
        for pct_missing in Y_train_pu.keys():
            for w_unl in np.arange(0.1, 1.01, 0.1):
                for clsf in ['mlp', 'hardb_mlp', 'clsdep_mlp']:
                    data = {}
                    data["iter"] = i
                    data["X_train"] = X_train
                    data["Y_train"] = Y_train_pu[pct_missing]
                    data["X_test"] = X_test
                    data["y_test"] = y_test
                    data["pct_missing"] = pct_missing
                    data["clsf"] = clsf
                    data["w_unlabelled"] = w_unl
                    args_list.append(data)

        # Start jobs
        pool = Pool(processes=8)
        acc_grid_list = pool.map(fit_classifier, args_list)
        acc_grid = pd.DataFrame(acc_grid_list)
        acc_grid.to_csv("weighted_unlabelled_moons.csv", index=False, mode='a')

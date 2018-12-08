from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier


def class_metrics(y_hat, y):
    accuracy = []
    f1_micro, f1_macro = [], []
    recall_micro, recall_macro = [], []

    if y_hat.shape != y.shape:
        raise Exception('Incompatible shape')
    for k in range(y_hat.shape[1]):
        recall_micro.append(recall_score(y_hat[:, k], y[:, k], average='micro'))
        recall_macro.append(recall_score(y_hat[:, k], y[:, k], average='macro'))

        f1_micro.append(f1_score(y_hat[:, k], y[:, k], average='micro'))
        f1_macro.append(f1_score(y_hat[:, k], y[:, k], average='macro'))

        accuracy.append(accuracy_score(y_hat[:, k], y[:, k]))
    return {
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "accuracy": accuracy
    }


def classify(method, h_features, h_labels, val_iter=10):
    print('X shape:', h_features.shape)
    print('y shape:', h_labels.shape)

    print('Training -> {0} - Classifier will run {1} times'.format(str(method),
                                                                   val_iter))

    accuracy = []
    class_stats = []
    f1_micro, f1_macro = [], []
    recall_micro, recall_macro = [], []
    precision_micro, precision_macro = [], []

    for iter_idx in range(val_iter):
        print('run  - - - - - - - - -  {0} at: {1} '.format(iter_idx + 1,
                                                            datetime.now()))

        X_train, X_test, y_train, y_test = train_test_split(h_features,
                                                            h_labels,
                                                            test_size=0.2)
        classifier = MultiOutputClassifier(method)
        classifier.fit(X_train, y_train)
        y_hat = classifier.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_hat))
        f1_micro.append(f1_score(y_test, y_hat, average='micro'))
        f1_macro.append(f1_score(y_test, y_hat, average='macro'))
        recall_micro.append(recall_score(y_test, y_hat, average='micro'))
        recall_macro.append(recall_score(y_test, y_hat, average='macro'))
        precision_micro.append(precision_score(y_test, y_hat, average='micro'))
        precision_macro.append(precision_score(y_test, y_hat, average='macro'))

        class_stats.append(class_metrics(y_hat, y_test))

    return {
        "classwise_stats": class_stats,
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
    }

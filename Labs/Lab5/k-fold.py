import sklearn.datasets
import numpy as np
import MLandPattern.MLandPattern as ML
import scipy


def load_iris():
    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )
    return D, L


def k_fold(k, attributes, labels, previous_prob, model=""):
    section_size = int(attributes.shape[1] / k)
    cont = 0
    low = 0
    high = section_size
    final_acc = -1
    model = model.lower()
    for i in range(k):
        if not i:
            validation_att = attributes[:, low:high]
            validation_labels = labels[low:high]
            train_att = attributes[:, high:]
            train_labels = labels[high:]
            if model == "bayes":
                [S, _, acc] = ML.Naive_log_classifier(
                    train_att,
                    train_labels,
                    validation_att,
                    previous_prob,
                    validation_labels,
                )
            elif model == "tied bayes":
                [S, _, acc] = ML.Tied_Naive_classifier(
                    train_att,
                    train_labels,
                    validation_att,
                    previous_prob,
                    validation_labels,
                )
            else:
                [S, _, acc] = ML.MVG_log_classifier(
                    train_att,
                    train_labels,
                    validation_att,
                    previous_prob,
                    validation_labels,
                )
            final_acc = acc
            final_S = S
            continue
        low += section_size
        high += section_size
        if high > attributes.shape[1]:
            high = attributes.shape
        validation_att = attributes[:, low:high]
        validation_labels = labels[low:high]
        train_att = attributes[:, :low]
        train_labels = labels[:low]
        train_att = np.hstack((train_att, attributes[:, high:]))
        train_labels = np.hstack((train_labels, labels[high:]))
        if model == "bayes":
            [S, _, acc] = ML.Naive_log_classifier(
                train_att,
                train_labels,
                validation_att,
                previous_prob,
                validation_labels,
            )
        elif model == "tied bayes":
            [S, _, acc] = ML.Tied_Naive_classifier(
                train_att,
                train_labels,
                validation_att,
                previous_prob,
                validation_labels,
            )
        else:
            [S, _, acc] = ML.MVG_log_classifier(
                train_att,
                train_labels,
                validation_att,
                previous_prob,
                validation_labels,
            )
        final_acc += acc
        final_S += S
    final_acc /= k
    final_S /= k
    return (final_acc, final_S)


if __name__ == "__main__":
    [D, L] = load_iris()
    [accuracy, S] = k_fold(150, D, L, 1 / 3, "tied bayes")
    print(f"Defined Error {1 - accuracy}")

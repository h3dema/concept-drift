#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.naive_bayes import GaussianNB

from classifier.detector_classifier import DetectorClassifier
from concept_drift.adwin import AdWin
from concept_drift.page_hinkley import PageHinkley
from evaluation.prequential import prequential


def calculate_drift(X, y,
                    n_train=1000,
                    w=100,
                    lambda_=50,
                    clfs_label=["GaussianNB", "Page-Hinkley", "AdWin"],
                    ):
    result = dict()
    result['clfs'] = dict(zip(tuple(clfs_label), np.zeros(len(clfs_label))))
    result['y'] = y
    clfs = []
    if "GaussianNB" in clfs_label:
        clfs.append(GaussianNB())
    if "Page-Hinkley" in clfs_label:
        clfs.append(DetectorClassifier(GaussianNB(), PageHinkley(lambda_=lambda_), np.unique(y)))
    if "AdWin" in clfs_label:
        clfs.append(DetectorClassifier(GaussianNB(), AdWin(), np.unique(y)))

    for i in range(len(clfs)):
        with np.errstate(divide='ignore', invalid='ignore'):
            y_pre, time = prequential(X, y, clfs[i], n_train)

        estimator = (y[n_train:] == y_pre) * 1
        acc_run = np.convolve(estimator, np.ones((w,)) / w, 'same')
        result['clfs'][clfs_label[i]] = {'mean': np.mean(acc_run), 'w': w}

        if clfs[i].__class__.__name__ == "DetectorClassifier":
            acc = [acc_run[d] for d in clfs[i].detected_elements]
            points = [(x, y) for x, y in zip(clfs[i].detected_elements, acc)]
            result['clfs'][clfs_label[i]].update({'num': clfs[i].change_detected, 'points': points})

    return result


def plot_drift(data,
               print_drifts=False,
               plot_circles=["AdWin"]):

    plt.title("Accuracy (exact match)")
    plt.xlabel("Instances")
    plt.ylabel("Accuracy")

    y = data['y']

    ax = plt.gca()
    y_max = y.shape[0]
    ax.set_xlim((0, y_max))
    ax.set_ylim((0, 1))

    ellipse_y = 0.05
    ellipse_x = ellipse_y * y_max / 2
    ellipse_color = {"GaussianNB": 'blue', "Page-Hinkley": 'orange', "AdWin": 'green'}

    has_to_plot = len(plot_circles) > 0

    clfs = data['clfs']

    for clfs_label in clfs:
        print("\n{}:".format(clfs_label))
        print("Mean acc within the window {}: {}".format(data['clfs'][clfs_label]['w'],
                                                         data['clfs'][clfs_label]['mean']))

        if clfs_label in ["Page-Hinkley", "AdWin"]:
            n = data['clfs'][clfs_label]['num']
            print("Drift detection: {}".format(n))

            points = data['clfs'][clfs_label].get('points', [])
            if print_drifts and n > 0:
                print("Drift detected in", str(points))

            if has_to_plot and clfs_label in plot_circles:
                for x, y, in points:
                    # c = plt.Circle((x, y), 0.1, color='r')
                    c = Ellipse(xy=(x, y), width=ellipse_x, height=ellipse_y, angle=0,
                                color=ellipse_color[clfs_label],
                                clip_on=False,
                                fill=False,
                                )
                    ax.add_artist(c)

        acc_run = [y for x, y in data]
        plt.plot(acc_run, "-", label=clfs_label, color=ellipse_color[clfs_label])

    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.show()

import os
import csv
import h5py
import pickle
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from texttable import Texttable
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

# region Global Variables

modalities_scores = dict()
starting_weights = list()
best_weights = list()
train_labels = None
test_labels = None
val_labels = None
best_accuracy_1k = -np.inf
step = 0.1

# endregion

# region Loading


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ntu/xsub', choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset'},
                        required=True, help='the work folder for storing results')
    parser.add_argument('--mod1', default=None, help='Directory containing "epoch1_test_score.pkl" for mod1 eval results')
    parser.add_argument('--mod2', default=None)
    parser.add_argument('--mod3', default=None)
    parser.add_argument('--mod4', default=None)
    parser.add_argument('--mod5', default=None)
    parser.add_argument('--mod6', default=None)
    return parser.parse_args()


def load_ground_truth_labels():
    global test_labels, val_labels, train_labels

    if 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = h5py.File('./data/' + 'ntu120/' + 'NTU_Csub.h5', 'r')
            train_labels = np.argmax(npz_data['y'][:], -1)
            test_labels = np.argmax(npz_data['test_y'][:], -1)
            val_labels = np.argmax(npz_data['valid_y'][:], -1)
        elif 'xset' in arg.dataset:
            npz_data = h5py.File('./data/' + 'ntu120/' + 'NTU_CSet.h5', 'r')
            train_labels = np.argmax(npz_data['y'][:], -1)
            test_labels = np.argmax(npz_data['test_y'][:], -1)
            val_labels = np.argmax(npz_data['valid_y'][:], -1)
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = h5py.File('./data/' + 'ntu/' + 'NTU_CS.h5', 'r')
            train_labels = np.argmax(npz_data['y'][:], -1)
            test_labels = np.argmax(npz_data['test_y'][:], -1)
            val_labels = np.argmax(npz_data['valid_y'][:], -1)
        elif 'xview' in arg.dataset:
            npz_data = h5py.File('./data/' + 'ntu/' + 'NTU_CV.h5', 'r')
            train_labels = np.argmax(npz_data['y'][:], -1)
            test_labels = np.argmax(npz_data['test_y'][:], -1)
            val_labels = np.argmax(npz_data['valid_y'][:], -1)
    else:
        raise NotImplementedError("only ntu120 and ntu datasets are supported")


def load_modalities_scores(given_paths, use_softmax=False):
    # scores_list a list with a len equal to the number of actions (2005 for val) / scores_list[0] is a tuple with 2 items,
    # the first is the name of the sequence like 'val0' and the second one is a ndarray with 60 values (prediction for each class).
    for name, path in given_paths.items():
        if path is not None:
            scores = dict()
            with open(os.path.join(path, 'epoch1_test_score.pkl'), 'rb') as test_scores:
                scores_list = (list(pickle.load(test_scores).items()))
                numpy_data = np.array([k[1] for k in scores_list])  # get rid of the names and convert to numpy data
                scores['test'] = softmax(numpy_data, axis=1) if use_softmax else numpy_data
            with open(os.path.join(path, 'epoch1_val_score.pkl'), 'rb') as val_scores:
                scores_list = (list(pickle.load(val_scores).items()))
                numpy_data = np.array([k[1] for k in scores_list])  # get rid of the names and convert to numpy data
                scores['val'] = softmax(numpy_data, axis=1) if use_softmax else numpy_data
            modalities_scores[name] = scores

# endregion

# region Accuracy


def calculate_accuracy(given_weights, mode, show_progress=False):
    assert len(given_weights) != 0, "No weights where given!!!"
    right_num = total_num = right_num_5 = 0
    prediction_labels = list()
    labels = val_labels if mode == 'val' else test_labels
    iterator = tqdm(range(len(labels)), ncols=100) if show_progress else range(len(labels))
    # for each sequence get the fusion score of the different modalities
    for action_index in iterator:
        label = labels[action_index]
        fused_scores = None
        for score_index, score in enumerate(modalities_scores.values()):
            score_value = score[mode][action_index] * given_weights[score_index]
            fused_scores = fused_scores + score_value if fused_scores is not None else score_value
        rank_5 = fused_scores.argsort()[-5:]
        right_num_5 += int(int(label) in rank_5)
        fused_prediction = np.argmax(fused_scores)
        prediction_labels.append(fused_prediction)
        right_num += int(fused_prediction == int(label))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    return acc, acc5, prediction_labels


def find_best_weights(given_counters, mode):
    weights_temp = [j for j in given_counters]
    accuracy_1k, _, _ = calculate_accuracy(weights_temp, mode=mode, show_progress=False)
    global best_accuracy_1k, best_weights
    if accuracy_1k > best_accuracy_1k:
        best_accuracy_1k = accuracy_1k
        best_weights = weights_temp


def recursive_loop(counters, length, level=0, show_progress=False):
    if level == len(counters):
        find_best_weights(counters, mode='val')
    else:
        counters[level] = starting_weights[level]
        iterator = tqdm(np.arange(counters[level], length[level] + 0.001, step), ncols=100) if show_progress \
            else np.arange(counters[level], length[level], step)
        for i in iterator:
            counters[level] = i
            recursive_loop(counters, length, level + 1)


def grid_search(given_step=0.1, max_weight_value=1, given_weights=None, region=0.1):
    if given_weights is not None:
        assert (len(given_weights) == len(modalities_scores))

        lower_bounds = [i - region for i in given_weights]
        upper_bounds = [i + region for i in given_weights]
    else:
        lower_bounds = [given_step for i in range(len(modalities_scores))]
        upper_bounds = [max_weight_value for i in range(len(modalities_scores))]

    global starting_weights, step
    step = given_step
    starting_weights = list.copy(lower_bounds)
    recursive_loop(lower_bounds, upper_bounds, show_progress=True)

# endregion


def get_results(given_step, max_weight_value=1, mode='test'):

    grid_search(given_step=given_step, max_weight_value=max_weight_value)  # finds the best weights
    test_acc, test_acc5, pred_labels = calculate_accuracy(best_weights, mode=mode, show_progress=True)

    print(best_weights)
    print('Top1 Acc: {:.3f}%'.format(test_acc * 100))
    print('Top5 Acc: {:.3f}%'.format(test_acc5 * 100))

    # grid_search(given_step=given_step * 0.2, max_weight_value=max_weight_value, given_weights=best_weights)
    # test_acc, test_acc5, pred_labels = calculate_accuracy(best_weights, mode=mode, show_progress=True)
    #
    # print(best_weights)
    # print('Top1 Acc: {:.4f}%'.format(test_acc * 100))
    # print('Top5 Acc: {:.4f}%'.format(test_acc5 * 100))


if __name__ == "__main__":
    arg = get_arguments()
    load_ground_truth_labels()
    paths = {'mod1': arg.mod1, 'mod2': arg.mod2, 'mod3': arg.mod3, 'mod4': arg.mod4, 'mod5': arg.mod5, 'mod6': arg.mod6}
    load_modalities_scores(paths, use_softmax=False)

    get_results(given_step=0.1, max_weight_value=1, mode='val')

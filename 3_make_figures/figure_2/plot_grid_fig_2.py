# Save best parameters (global fit, Odoemene) for initializing individual fits
import numpy as np
import json
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, calculate_state_permutation, partition_data_by_session, create_violation_mask
from io_utils import load_cv_arr, load_glmhmm_data, load_data, load_session_fold_lookup, load_animal_list
import matplotlib.pyplot as plt

import psytrack as psy
import matplotlib
matplotlib.font_manager._rebuild()
colors = psy.COLORS
zorder = psy.ZORDER
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model

def load_hessian_data(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    hessian = data[0]
    std_dev = data[1]
    return hessian, std_dev



if __name__ == '__main__':
    data_dir = 'data/ibl/data_for_cluster/data_by_animal/'
    figure_dir = 'figures/figures_for_paper/figure_2/'

    animal = 'CSHL_008'

    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

    covar_set = 2
    alpha_val = 2
    sigma_val = 2
    K = 3
    D, M, C = 1, 3, 2

    results_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
        covar_set) + '/prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'


    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)


    # Get cvbt and pred acc:
    cols = ['#999999', '#984ea3', '#e41a1c', '#dede00']
    cv_arr = load_cv_arr(results_dir + "/cvbt_folds_model.npz")

    cv_arr_for_plotting = cv_arr[[0, 2, 3, 4, 5, 6], :]

    pred_acc_arr = load_cv_arr(results_dir + "predictive_accuracy_mat.npz")
    pred_acc_arr_for_plotting = pred_acc_arr[[0, 2, 3, 4, 5, 6], :]

    ## Plot cvbt and pred acc:
    fig = plt.figure(figsize=(5.5, 3))
    gs = fig.add_gridspec(2, 9)
    ax1 = fig.add_subplot(gs[0, 3:6])

    plt.subplots_adjust(wspace=5.5, hspace=0.5)
    plt.plot([0, 0.5, 1, 2, 3, 4], np.mean(cv_arr_for_plotting, axis=1), '-o', color=cols[0], zorder=0, alpha=1,
             lw=1.5, markersize=4)
    plt.yticks([0.30, 0.35, 0.4, 0.45], labels=["0.30", "0.35", "0.40", "0.45"], fontsize=10)
    plt.xticks([0, 0.5, 1, 2, 3, 4], labels=['1', 'L.', '2', '3', '4', '5'], fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.ylabel("test LL (bits/trial)", fontsize=10)
    plt.xlabel("# states", fontsize=10)


    fig.add_subplot(gs[0, 6:9])
    plt.plot([0, 0.5, 1, 2, 3, 4], np.mean(pred_acc_arr_for_plotting, axis=1), '-o', color=cols[1], zorder=0, alpha=1,
             lw=1.5, markersize=4)
    plt.yticks([0.78, 0.8, 0.82, 0.84], labels=["78", "80", "82", "84"], fontsize=10)
    plt.xticks([0, 0.5, 1, 2, 3, 4], labels=['1', 'L.', '2', '3', '4', '5'], fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.ylabel("predictive acc. (%)", fontsize=10)
    plt.xlabel("# states", fontsize=10)


    # Get the file name corresponding to the best initialization for given K value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    transition_matrix = np.exp(hmm_params[1][0])
    weight_vectors = -hmm_params[2]

    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    # Plot these too:
    fig.add_subplot(gs[1, 3:6])

    M = weight_vectors.shape[2] - 1
    for k in range(K):
        plt.plot(range(M + 1), weight_vectors[k][0][[0, 3, 1, 2]], marker='o', label= "state " + str(k+1), color=cols[k],
                 lw=1, alpha = 0.7)
    plt.yticks(fontsize=10)
    plt.xticks([0, 1, 2, 3], ['stimulus', 'bias', 'prev. \nchoice', 'win-stay-\nlose-switch'], fontsize = 10, rotation = 45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--", lw = 0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    ax = plt.gca()

    plt.ylabel("GLM weight", fontsize = 10)
    fig.savefig(figure_dir + 'fig2_grid_v2.pdf')

    fig = plt.figure(figsize=(1, 1.2))
    accuracies = [80, 90, 60, 58]
    for z, occ in enumerate(accuracies):
        if z ==0:
            col ='grey'
        else:
            col =cols[z-1]
        plt.bar(z, occ, width = 0.8, color = col)
    plt.ylim((50, 100))
    plt.xticks([0, 1, 2,3], ['All','1', '2', '3'], fontsize = 10)
    plt.yticks([50, 75, 100], fontsize=10)
    plt.xlabel('state', fontsize = 10)
    plt.ylabel('accuracy (%)', fontsize=10,labelpad=-0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(figure_dir + 'acc_in_state.pdf')

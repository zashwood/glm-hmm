# Produce figures 7d,e,f for human/Urai et al. (2017) dataset
# Note: assumes that you have already run preprocessing and model fitting on
# this dataset
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('../')

from plotting_utils import load_glmhmm_data, load_cv_arr, \
    load_data, load_animal_list, \
    get_file_name_for_best_model_fold, \
    partition_data_by_session, create_violation_mask, \
    get_marginal_posterior, get_global_weights, get_global_trans_mat, \
    get_prob_right


if __name__ == '__main__':

    figure_dir = '../../figures/figure_7/'

    outer_data_dir = '../../data/human/data_for_cluster/'
    overall_dir = '../../results/human_individual_fit/'
    global_results_dir = '../../results/human_global_fit/'

    data_dir = outer_data_dir + 'data_by_animal/'

    animal_list = load_animal_list(data_dir + 'animal_list.npz')

    cols = [
        '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999',
        '#e41a1c', '#dede00'
    ]

    K = 2
    D, M, C = 1, 3, 2

    global_weights = get_global_weights(global_results_dir, K)
    fig = plt.figure(figsize=(6.2, 4.5))
    plt.subplots_adjust(left=0.1,
                        bottom=0.2,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.6)
    for k in range(K):
        plt.subplot(2, 3, k + 1)

        for animal in animal_list:
            results_dir = overall_dir + animal + '/'

            cv_file = results_dir + "/cvbt_folds_model.npz"
            cvbt_folds_model = load_cv_arr(cv_file)

            with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
                best_init_cvbt_dict = json.load(f)

            # Get the file name corresponding to the best initialization for
            # given K value
            raw_file = get_file_name_for_best_model_fold(
                cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)

            transition_matrix = np.exp(hmm_params[1][0])
            weight_vectors = -hmm_params[2]

            plt.plot(range(M + 1),
                     weight_vectors[k][0][[0, 3, 1, 2]],
                     '-o',
                     color=cols[k],
                     lw=1,
                     alpha=0.7,
                     markersize=3)
        if k == 0:
            plt.yticks([-2, 0, 2, 4], fontsize=10)
            plt.xticks([0, 1, 2, 3], ['stim.', 'bias', 'p.c.', 'w.s.l.s.'],
                       fontsize=10,
                       rotation=20)
            plt.ylabel("GLM weight", fontsize=10)
        else:
            plt.yticks([-2, 0, 2, 4], ['', '', '', ''])
            plt.xticks([0, 1, 2, 3], ['', '', '', ''])
        plt.title("state " + str(k + 1), fontsize=10, color=cols[k])
        plt.plot(range(M + 1),
                 global_weights[k][0][[0, 3, 1, 2]],
                 '-o',
                 color='k',
                 lw=1.3,
                 alpha=1,
                 markersize=3,
                 label='global')
        plt.axhline(y=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
        if k == 1:
            plt.legend(fontsize=10,
                       labelspacing=0.2,
                       handlelength=1,
                       borderaxespad=0.2,
                       borderpad=0.2,
                       framealpha=0,
                       bbox_to_anchor=(0.53, 1),
                       handletextpad=0.2,
                       loc='upper left')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.ylim((-2, 4.9))

    # ==================== GLM curves ==================
    inpt, old_y, session = load_data(outer_data_dir + 'all_animals_concat.npz')
    unnormalized_inpt, _, _ = load_data(outer_data_dir +
                                        'all_animals_concat_unnormalized.npz')
    y = np.copy(old_y)

    # Restrict to non-violation trials:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, train_masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

    # Save parameters for initializing individual fits
    covar_set = 2
    # Get posterior probs:
    results_dir = global_results_dir
    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K
    # value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                 results_dir,
                                                 best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                             hmm_params, K, range(K))
    _, counts = np.unique(np.argmax(posterior_probs, axis=1),
                          return_counts=True)

    cols = [
        '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999',
        '#e41a1c', '#dede00'
    ]
    labels = ["'biased left'", "'biased right'"]
    plt.subplot(2, 3, 3)
    for k in range(K):
        # Get index of trials where posterior_probs > 0.9 and not violation
        idx_of_interest = \
            np.where((posterior_probs[:, k] >= 0.9) & (mask == 1))[0]
        inpt_this_state, unnormalized_inpt_this_state, y_this_state = inpt[
                                                                      idx_of_interest,
                                                                      :], \
                                                                      unnormalized_inpt[
                                                                      idx_of_interest,
                                                                      :], \
                                                                      old_y[
                                                                          idx_of_interest,
                                                                          :]

        stim_vals, prob_right_max = get_prob_right(weight_vectors, inpt, k, 1,
                                                   1)
        _, prob_right_min = get_prob_right(weight_vectors, inpt, k, -1, -1)

        plt.plot(stim_vals,
                 prob_right_max,
                 '-',
                 color=cols[k],
                 alpha=1,
                 lw=1,
                 zorder=5)  # went R and was rewarded
        plt.plot(stim_vals,
                 get_prob_right(weight_vectors, inpt, k, -1, 1)[1],
                 '--',
                 color=cols[k],
                 alpha=0.5,
                 lw=1)  # went L and was not rewarded
        plt.plot(stim_vals,
                 get_prob_right(weight_vectors, inpt, k, 1, -1)[1],
                 '-',
                 color=cols[k],
                 alpha=0.5,
                 lw=1,
                 markersize=3)  # went R and was not rewarded
        plt.plot(stim_vals, prob_right_min, '--', color=cols[k], alpha=1,
                 lw=1)  # went L and was rewarded

        if k == 0:
            plt.ylabel("p(choice = 'more')", fontsize=10)
            plt.xlabel('sensory evidence (a.u.)', fontsize=10)
            plt.xticks([
                -1.8417554175746027, -0.014912144544934091, 1.8119311284847344
            ],
                       labels=['-5', '0', '5'],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], labels=['0', '0.5', '1'], fontsize=10)
        else:
            plt.xticks([
                -1.8417554175746027, -0.014912144544934091, 1.8119311284847344
            ],
                       labels=['', '', ''],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], labels=['', '', ''], fontsize=10)
            plt.ylabel('')
            plt.xlabel('')
        plt.ylim((0, 1))
        plt.axhline(y=0.5, color="k", alpha=0.5, ls="--", linewidth=0.75)
        plt.axvline(x=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
    plt.ylabel("p(choice = 'more')", fontsize=10)
    plt.xlabel('sensory evidence (a.u.)', fontsize=10)
    plt.xticks(
        [-1.8417554175746027, -0.014912144544934091, 1.8119311284847344],
        labels=['-5', '0', '-5'],
        fontsize=10)
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
    plt.ylim((-0.05, 1.05))
    plt.axhline(y=0.5, color="k", alpha=0.5, ls="--", linewidth=0.75)
    plt.axvline(x=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # ================ DWELL TIMES ==================================
    global_transition_matrix = get_global_trans_mat(global_results_dir, K)
    global_dwell_times = 1 / (np.ones(K) - global_transition_matrix.diagonal())

    state_dwell_times = np.zeros((len(animal_list), K))
    for k in range(K):
        plt.subplot(2, 3, 4 + k)
        for z, animal in enumerate(animal_list):
            results_dir = overall_dir + animal + '/'

            cv_file = results_dir + "/cvbt_folds_model.npz"
            cvbt_folds_model = load_cv_arr(cv_file)

            with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
                best_init_cvbt_dict = json.load(f)

            # Get the file name corresponding to the best initialization for
            # given K value
            raw_file = get_file_name_for_best_model_fold(
                cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)

            transition_matrix = np.exp(hmm_params[1][0])

            state_dwell_times[z, :] = 1 / (np.ones(K) -
                                           transition_matrix.diagonal())

        logbins = np.logspace(np.log10(1),
                              np.log10(max(state_dwell_times[:, k])), 15)
        plt.hist(state_dwell_times[:, k],
                 bins=logbins,
                 color=cols[k],
                 histtype='bar',
                 rwidth=0.8)
        ax1 = plt.gca()
        ax1.set_xscale('log')
        ax1.set_xticks([1, 10, 100, 1000])
        ax1.set_yticks([0, 2, 4, 6])
        plt.ylim((0, 6))
        plt.axvline(np.median(state_dwell_times[:, k]),
                    linestyle='--',
                    color='k',
                    lw=1,
                    label='median')
        ax1.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        if k == 0:
            ax1.set_yticklabels(labels=["0", "2", "4", "6"],
                                fontsize=10,
                                alpha=1)
            ax1.set_xticklabels(labels=["1", "10", "100", "1000"],
                                fontsize=10,
                                alpha=1,
                                rotation=45)
            plt.ylabel("# participants", fontsize=10)
            plt.xlabel("expected dwell time \n (# trials)",
                       fontsize=10,
                       labelpad=0)
        else:
            ax1.set_yticklabels(labels=["", "", "", ""], fontsize=10, alpha=1)
            ax1.set_xticklabels(labels=["", "", "", ""],
                                fontsize=10,
                                alpha=1,
                                rotation=45)
        if k == 1:
            plt.legend(fontsize=10,
                       labelspacing=0.2,
                       handlelength=0.8,
                       borderaxespad=0.2,
                       borderpad=0.2,
                       framealpha=0,
                       bbox_to_anchor=(-0.01, 1.12),
                       handletextpad=0.2,
                       loc='upper left')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    fig.savefig(figure_dir + 'fig7.pdf')

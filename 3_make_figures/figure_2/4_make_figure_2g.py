# Plot figure 2g of Ashwood et al. (2020)
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, \
    partition_data_by_session, create_violation_mask, \
    get_marginal_posterior, get_prob_right


if __name__ == '__main__':
    animal = "CSHL_008"
    alpha_val = 2
    sigma_val = 2

    data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    results_dir = '../../results/ibl_individual_fit/' + animal + '/'
    figure_dir = '../../figures/figure_2/'

    fig = plt.figure(figsize=(4.6, 2), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.13, bottom=0.23, right=0.9, top=0.8)
    K = 3
    inpt, y, session = load_data(data_dir + animal + '_processed.npz')
    unnormalized_inpt, _, _ = load_data \
        (data_dir + animal + '_unnormalized.npz')

    # Create masks for violation trials
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

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
    weight_vectors = hmm_params[2]

    posterior_probs = get_marginal_posterior(inputs, datas, masks,
                                             hmm_params, K, range(K))
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    cols = [
        '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00'
    ]
    for k in range(K):
        plt.subplot(1, 3, k+1)
        # USE GLM WEIGHTS TO GET PROB RIGHT
        stim_vals, prob_right_max = get_prob_right(-weight_vectors, inpt, k, 1,
                                                   1)
        _, prob_right_min = get_prob_right(-weight_vectors, inpt, k, -1, -1)
        plt.plot(stim_vals,
                 prob_right_max,
                 '-',
                 color=cols[k],
                 alpha=1,
                 lw=1,
                 zorder=5)  # went R and was rewarded on previous trial
        plt.plot(stim_vals,
                 get_prob_right(-weight_vectors, inpt, k, -1, 1)[1],
                 '--',
                 color=cols[k],
                 alpha=0.5,
                 lw=1)  # went L and was not rewarded on previous trial
        plt.plot(stim_vals,
                 get_prob_right(-weight_vectors, inpt, k, 1, -1)[1],
                 '-',
                 color=cols[k],
                 alpha=0.5,
                 lw=1,
                 markersize=3)  # went R and was not rewarded on previous trial
        plt.plot(stim_vals, prob_right_min, '--', color=cols[k], alpha=1,
                 lw=1)  # went L and was rewarded on previous trial
        plt.xticks([min(stim_vals), 0, max(stim_vals)],
                   labels=['', '', ''],
                   fontsize=10)
        plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)
        plt.ylabel('')
        plt.xlabel('')
        if k == 0:
            plt.title("state 1 \n(\"engaged\")", fontsize=10, color=cols[k])
            plt.xticks([min(stim_vals), 0, max(stim_vals)],
                       labels=['-100', '0', '100'],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
            plt.ylabel('p("R")', fontsize=10)
            plt.xlabel('stimulus', fontsize=10)
        if k == 1:
            plt.title("state 2 \n(\"biased left\")",
                      fontsize=10,
                      color=cols[k])
            plt.xticks([min(stim_vals), 0, max(stim_vals)],
                       labels=['', '', ''],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)
        if k == 2:
            plt.title("state 3 \n(\"biased right\")",
                      fontsize=10,
                      color=cols[k])
            plt.xticks([min(stim_vals), 0, max(stim_vals)],
                       labels=['', '', ''],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], ['', '', ''], fontsize=10)
        plt.axhline(y=0.5, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.axvline(x=0, color="k", alpha=0.45, ls=":", linewidth=0.5)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.ylim((-0.01, 1.01))
    fig.savefig(figure_dir + 'fig2g.pdf')

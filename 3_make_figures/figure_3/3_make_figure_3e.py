# Plot number of state changes in 90 trials (Figure 3e of Ashwood et al. (
# 2020))
import json
import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, find_change_points

if __name__ == '__main__':
    animal = "CSHL_008"
    K = 3

    data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    results_dir = '../../results/ibl_individual_fit/' + animal + '/'
    figure_dir = '../../figures/figure_3/'

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

    # Save parameters for initializing individual fits
    weight_vectors = hmm_params[2]
    log_transition_matrix = hmm_params[1][0]
    init_state_dist = hmm_params[0][0]

    # Also get data for animal:
    inpt, y, session = load_data(data_dir + animal + '_processed.npz')
    all_sessions = np.unique(session)

    # Create mask:
    # Identify violations for exclusion:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

    posterior_probs = [get_marginal_posterior([input], [data], [mask],
                                             hmm_params, K, range(K)) for
                       input, data, mask in zip(inputs, datas, masks)]

    states_max_posterior = [
        np.argmax(posterior_prob, axis=1) for posterior_prob in posterior_probs
    ]  # list of states at each trial in session

    change_points = find_change_points(states_max_posterior)
    num_sess = len(change_points)
    change_points_per_sess = []
    for sess in range(num_sess):
        change_points_per_sess.append(len(change_points[sess]))

    cp_bin_locs, cp_hist = np.unique(change_points_per_sess,
                                     return_counts=True)

    fig = plt.figure(figsize=(2, 2))
    plt.subplots_adjust(left=0.4, bottom=0.3, right=0.95, top=0.95)
    frac_non_zero = 0
    for z, occ in enumerate(cp_hist):
        plt.bar(cp_bin_locs[z],
                occ / num_sess,
                width=0.8,
                color=(66 / 255, 150 / 255, 129 / 255))
        if z > 0:
            frac_non_zero += occ / num_sess

    plt.ylim((0, 0.3))
    plt.xticks([0, 1, 2, 3, 4, 5, 6], fontsize=10)
    plt.yticks([0, 0.15, 0.3], fontsize=10)
    plt.xlabel('# state changes \n in 90 trials', fontsize=10)
    plt.ylabel('frac. sessions', fontsize=10)  # , labelpad=-0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir + 'fig3e.pdf')

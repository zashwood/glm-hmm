# Create Figure 3d - fractional occupancies of 3 states
import sys
import json

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    get_file_name_for_best_model_fold, partition_data_by_session, \
    create_violation_mask, get_marginal_posterior, get_was_correct

if __name__ == '__main__':
    animal = "CSHL_008"
    K = 3

    data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    results_dir = '../../results/ibl_individual_fit/' + animal + '/'
    figure_dir = '../../figures/figure_3/'

    inpt, y, session = load_data(data_dir + animal + '_processed.npz')
    unnormalized_inpt, _, _ = load_data(data_dir + animal +
                                        '_unnormalized.npz')

    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

    # get params for posterior probs
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

    posterior_probs = get_marginal_posterior(inputs, datas, masks,
                                             hmm_params, K, range(K))

    states_max_posterior = np.argmax(posterior_probs, axis=1)
    state_occupancies = []
    cols = [
        '#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00'
    ]
    for k in range(K):
        # Get state occupancy:
        occ = len(
            np.where(states_max_posterior == k)[0]) / len(states_max_posterior)
        state_occupancies.append(occ)

    # ====================== PLOTTING CODE ===============================
    fig = plt.figure(figsize=(1.3, 1.7))
    plt.subplots_adjust(left=0.4, bottom=0.3, right=0.95, top=0.95)
    for z, occ in enumerate(state_occupancies):
        plt.bar(z, occ, width=0.8, color=cols[z])
    plt.ylim((0, 1))
    plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize=10)
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
    plt.xlabel('state', fontsize=10)
    plt.ylabel('frac. occupancy', fontsize=10)  #, labelpad=0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir + 'fig3d.pdf')

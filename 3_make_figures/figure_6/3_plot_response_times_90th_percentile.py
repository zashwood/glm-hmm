# plot differences in 90th percentile response times for engaged and
# disengaged states for IBL animals (figure 6b)
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    load_animal_list, load_rts, get_file_name_for_best_model_fold, \
    partition_data_by_session, create_violation_mask, \
    get_marginal_posterior, calculate_state_permutation, \
    perform_bootstrap_individual_animal, read_bootstrapped_median


if __name__ == '__main__':
    data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    response_time_dir = '../../data/ibl/response_times/data_by_animal/'
    overall_dir = '../../results/ibl_individual_fit/'
    figure_dir = '../../figures/figure_6/'
    animal_list = load_animal_list(data_dir + 'animal_list.npz')
    animal_list = animal_list[[[
        21, 35, 18, 19, 26, 11, 16, 17, 24, 13, 22, 10, 20, 9, 31, 14, 23, 33,
        15, 8, 27, 1, 34, 25, 3, 32, 36, 4, 29, 30, 0, 6, 28, 2, 5, 12, 7
    ]]] # order animals by differences

    fig, ax = plt.subplots(figsize=(4, 6))
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.9, top=0.95)
    for z, animal in enumerate(animal_list):
        results_dir = overall_dir + animal

        cv_file = results_dir + "/cvbt_folds_model.npz"
        cvbt_folds_model = load_cv_arr(cv_file)

        K = 3
        with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
            best_init_cvbt_dict = json.load(f)

        raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                     results_dir,
                                                     best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)

        if animal == "ibl_witten_05" or animal == "CSHL_001":
            permutation = calculate_state_permutation(hmm_params)
        else:
            permutation = range(K)
        # Also get data for animal:
        inpt, y, session = load_data(data_dir + animal + '_processed.npz')
        # Create mask:
        # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                       inpt.shape[0])
        y[np.where(y == -1), :] = 1
        inputs, datas, train_masks = partition_data_by_session(
            np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

        posterior_probs = get_marginal_posterior(inputs, datas, train_masks,
                                                 hmm_params, K, permutation)
        # Read in RTs
        rts, rts_sess = load_rts(response_time_dir + animal + '.npz')

        rts_engaged = rts[np.where(posterior_probs[:, 0] >= 0.9)[0]]
        rts_engaged = rts_engaged[np.where(~np.isnan(rts_engaged))]
        rts_disengaged = rts[np.where((posterior_probs[:, 1] >= 0.9)
                                      | (posterior_probs[:, 2] >= 0.9))[0]]
        rts_disengaged = rts_disengaged[np.where(~np.isnan(rts_disengaged))]

        # Get 90th percentile for each
        quant = 0.90
        eng_quantile = np.quantile(rts_engaged, quant)
        dis_quantile = np.quantile(rts_disengaged, quant)
        diff_quantile = dis_quantile - eng_quantile

        # Perform bootstrap to get error bars:
        lower_eng, upper_eng, min_val_eng, max_val_eng, frac_above_true \
            = perform_bootstrap_individual_animal(rts_engaged, rts_disengaged,
                                                  diff_quantile, quant)
        plt.scatter(diff_quantile, z, color='r', s=5)
        plt.plot([lower_eng, upper_eng], [z, z], color='r', lw=0.75)
    median, lower, upper, mean_viol_rate_dist = read_bootstrapped_median(
        overall_dir + 'median_response_bootstrap.npz')
    plt.plot([lower, upper], [z + 1, z + 1], color='#0343df', lw=0.75)
    plt.scatter(median, z + 1, color='b', s=1)
    plt.ylabel("animal (IBL)", fontsize=10)
    plt.xlabel(
        "$\Delta$ 90th quantile \nresponse time (s) \n (disengaged - " \
        "\nengaged)",
        fontsize=10)
    plt.xticks([0, 2.5, 5, 7.5, 10, 12.5, 15],
               ['0', '', '5', '', '10', '', '15'],
               fontsize=10)
    plt.axvline(x=0, linestyle='--', color='k', alpha=0.5, lw=0.75)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.show()
    fig.savefig(figure_dir + 'fig6b.pdf')

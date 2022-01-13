# Create bootstrap distribution for median response time across animals
# WARNING: this takes a while (1-2 hours)!
import json
import sys

import numpy as np

sys.path.append('../')
from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    load_animal_list, load_rts, get_file_name_for_best_model_fold, \
    partition_data_by_session, create_violation_mask, \
    get_marginal_posterior, calculate_state_permutation

np.random.seed(80)


def get_bootstrap_err_bars(overall_dir, response_time_dir, data_dir,
                           animal_list):
    # Get mean across all animals:
    median_response_time_dist = []

    for i in range(2000):
        dist_across_animals = []
        for z, animal in enumerate(animal_list):
            results_dir = overall_dir + animal + '/'

            cv_file = results_dir + "/cvbt_folds_model.npz"

            cvbt_folds_model = load_cv_arr(cv_file)

            K = 3
            with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
                best_init_cvbt_dict = json.load(f)

            # Get the file name corresponding to the best initialization for
            # given K value
            raw_file = get_file_name_for_best_model_fold(
                cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)

            # Also get data for animal:
            inpt, y, session = load_data(data_dir + animal + '_processed.npz')

            # Create mask:
            # Identify violations for exclusion:
            violation_idx = np.where(y == -1)[0]
            nonviolation_idx, mask = create_violation_mask(
                violation_idx, inpt.shape[0])
            y[np.where(y == -1), :] = 1
            inputs, datas, train_masks = partition_data_by_session(
                np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

            if animal == "ibl_witten_05" or animal == "CSHL_001":
                permutation = calculate_state_permutation(hmm_params)
            else:
                permutation = range(K)

            posterior_probs = get_marginal_posterior(inputs, datas,
                                                     train_masks, hmm_params,
                                                     K, permutation)
            locx_engaged = np.where(posterior_probs[:, 0] >= 0.9)[0]
            locx_disengaged = np.where((posterior_probs[:, 1] >= 0.9)
                                       | (posterior_probs[:, 2] >= 0.9))[0]

            # Sample locx_engaged
            sampled_locx_engaged = np.random.choice(locx_engaged,
                                                    replace=True,
                                                    size=len(locx_engaged))
            # Sample locx_disengaged
            sampled_locx_disengaged = np.random.choice(
                locx_disengaged, replace=True, size=len(locx_disengaged))

            # Read in RTs
            rts, rts_sess = load_rts(response_time_dir + animal + '.npz')

            rts_engaged = rts[sampled_locx_engaged]
            rts_engaged = rts_engaged[np.where(~np.isnan(rts_engaged))]
            rts_disengaged = rts[sampled_locx_disengaged]
            rts_disengaged = rts_disengaged[np.where(
                ~np.isnan(rts_disengaged))]

            # Get 90th percentile for each
            quant = 0.90
            eng_quantile = np.quantile(rts_engaged, quant)
            dis_quantile = np.quantile(rts_disengaged, quant)
            diff_quantile = dis_quantile - eng_quantile

            dist_across_animals.append(diff_quantile)
        print(np.median(np.array(dist_across_animals)))
        median_response_time_dist.append(
            np.median(np.array(dist_across_animals)))
    lower = np.quantile(np.array(median_response_time_dist), 0.025)
    upper = np.quantile(np.array(median_response_time_dist), 0.975)
    median = np.quantile(np.array(median_response_time_dist), 0.5)
    return median, lower, upper, median_response_time_dist


if __name__ == '__main__':
    data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    response_time_dir = '../../data/ibl/response_times/data_by_animal/'
    overall_dir = '../../results/ibl_individual_fit/'
    figure_dir = '../../figures/figure_6/'

    animal_list = load_animal_list(data_dir + 'animal_list.npz')
    median, lower, upper, median_response_time_dist = \
        get_bootstrap_err_bars(overall_dir, response_time_dir, data_dir,
                               animal_list)
    np.savez(overall_dir + 'median_response_bootstrap.npz', median, lower,
             upper, median_response_time_dist)

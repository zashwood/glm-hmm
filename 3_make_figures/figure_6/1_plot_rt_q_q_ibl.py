# Create Q-Q plot of response times for engaged and disengaged states (
# Figure 6a)
import json
import sys
import os

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')

from plotting_utils import load_glmhmm_data, load_cv_arr, load_data, \
    load_animal_list, load_rts, get_file_name_for_best_model_fold, \
    partition_data_by_session, create_violation_mask, \
    get_marginal_posterior, calculate_state_permutation

if __name__ == '__main__':
    cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"]

    data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    response_time_dir = '../../data/ibl/response_times/data_by_animal/'
    overall_dir = '../../results/ibl_individual_fit/'
    figure_dir = '../../figures/figure_6/'
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    animal_list = load_animal_list(data_dir + 'animal_list.npz')

    fig, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(left=0.3, bottom=0.3, right=0.9, top=0.9)
    for animal in animal_list:
        rts, rts_sess = load_rts(response_time_dir + animal + '.npz')

        results_dir = overall_dir + animal + '/'

        cv_file = results_dir + "/cvbt_folds_model.npz"
        cvbt_folds_model = load_cv_arr(cv_file)

        K = 3
        with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
            best_init_cvbt_dict = json.load(f)

        # Get the file name corresponding to the best initialization for
        # given K value
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

        # now loop through the sessions for which RTs exist:
        uniq_sessions = np.unique(rts_sess)
        for z, sess in enumerate(uniq_sessions):
            these_rts = rts[np.where(rts_sess==sess)]
            this_inpt, this_y, this_mask = inpt[np.where(session==sess)], \
                                           y[np.where(session==sess)], \
                                           mask[np.where(session==sess)]

            # append a column of 1s as bias:
            this_inpt = np.hstack((this_inpt, np.ones((len(this_inpt), 1))))
            posterior_probs = get_marginal_posterior([this_inpt], [this_y],
                                                     [this_mask],
                                                     hmm_params, K,
                                                     permutation)
            if z == 0:
                rts_engaged = these_rts[np.where(posterior_probs[:, 0] >= 0.9)[0]]
                rts_disengaged = these_rts[np.where((posterior_probs[:, 1] >= 0.9)
                                              | (posterior_probs[:, 2] >= 0.9))[0]]
            else:
                these_rts_engaged = these_rts[np.where(posterior_probs[:, 0] >= 0.9)[0]]
                these_rts_disengaged = these_rts[np.where((posterior_probs[:, 1] >= 0.9)
                                              | (posterior_probs[:, 2] >= 0.9))[0]]
                rts_engaged = np.concatenate((rts_engaged, these_rts_engaged))
                rts_disengaged = np.concatenate((rts_disengaged, these_rts_disengaged))


        max_val = np.nanmax(rts_engaged)
        max_val_2 = np.nanmax(rts_disengaged)
        max_val_overall = np.nanmax([max_val, max_val_2])

        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        eng_quantiles = []
        dis_quantiles = []
        for i in np.arange(0.01, 1.01, 0.01):
            eng_quantile = np.quantile(
                rts_engaged[np.where(~np.isnan(rts_engaged))], i)
            dis_quantile = np.quantile(
                rts_disengaged[np.where(~np.isnan(rts_disengaged))], i)
            eng_quantiles.append(eng_quantile)
            dis_quantiles.append(dis_quantile)
            if i == 0.9:
                plt.scatter(eng_quantile,
                            dis_quantile,
                            color='r',
                            zorder=2,
                            s=0.75)
        plt.plot(eng_quantiles,
                 dis_quantiles,
                 'o-',
                 color='grey',
                 alpha=0.3,
                 linewidth=0.5,
                 markersize=0.5,
                 zorder=0)

        ax.set_ylim(ymin=0.1)
        ax.set_xlim(xmin=0.1)
        plt.ylabel("disengaged response time (s)", fontsize=10, labelpad=0)
        plt.xlabel("engaged response time (s)", fontsize=10, labelpad=0)

        _, p = stats.ks_2samp(rts_engaged, rts_disengaged)
        if p >= 0.05:
            print(animal + " does not reject null")
    plt.plot(np.arange(0.01, max_val_overall, max_val_overall / 100),
             np.arange(0.01, max_val_overall, max_val_overall / 100),
             color='k',
             linestyle='--')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    fig.savefig(figure_dir + 'fig6a.pdf')

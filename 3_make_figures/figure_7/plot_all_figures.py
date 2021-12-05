# Save best parameters (global fit, Odoemene) for initializing individual fits
import numpy as np
import json
import pandas as pd
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, calculate_state_permutation, partition_data_by_session, create_violation_mask, get_marginal_posterior
from io_utils import load_cv_arr, load_glmhmm_data, load_data, load_session_fold_lookup, load_animal_list
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.special import expit, logit


matplotlib.font_manager._rebuild()
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model

def load_switches_data(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cp_hist, cp_bin_locs, num_sess_all_animals, change_points_per_sess = data[0], data[1], data[2], data[3]
    return cp_hist, cp_bin_locs, num_sess_all_animals, change_points_per_sess

def load_hessian_data(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    hessian = data[0]
    std_dev = data[1]
    return hessian, std_dev

def load_best_params(raw_file):
    container = np.load(raw_file, allow_pickle=True)
    data = [container[key] for key in container]
    hmm_params = data[0]
    return hmm_params

def get_global_weights(K):
    results_dir = 'data/human/data_for_cluster/best_params/'
    raw_file = results_dir + 'best_params_K_' + str(K) + '.npz'
    hmm_params = load_best_params(raw_file)
    global_weights = -hmm_params[2]
    return global_weights, hmm_params

def calculate_psychometric_stats(contrast, choice, uniq_contrast_vals):
    # summary stats - average psychfunc over observers
    num_trials_vec = []
    mean_choice_vec = []
    for contrast_val in uniq_contrast_vals:
        # get number of trials
        trials_this_contrast_idx = np.where(contrast == contrast_val)[0]
        num_trials = len(trials_this_contrast_idx)
        num_trials_vec.append(num_trials)
        mean_choice = np.sum(choice[trials_this_contrast_idx])/num_trials
        mean_choice_vec.append(mean_choice)
    df = pd.DataFrame({'signed_contrast':uniq_contrast_vals, 'ntrials':num_trials_vec, 'fraction':mean_choice_vec})
    return df


def get_global_trans_mat(K):
    results_dir = 'data/human/data_for_cluster/best_params/'
    raw_file = results_dir + 'best_params_K_' + str(K) + '.npz'
    hmm_params = load_best_params(raw_file)
    global_transition_matrix = np.exp(hmm_params[1][0])
    return global_transition_matrix

def get_prob_right(weight_vectors, inpt, k, pc, wsls):
    # stim vector
    min_val_stim = np.min(inpt[:,0])
    max_val_stim = np.max(inpt[:,0])
    stim_vals = np.arange(min_val_stim, max_val_stim, 0.05)
    x = np.array([stim_vals, np.repeat(pc, len(stim_vals)), np.repeat(wsls, len(stim_vals)), np.repeat(1, len(stim_vals))]).T
    wx = np.matmul(x, weight_vectors[k][0])
    return stim_vals, expit(wx)

# Bin click diffs into groups of 3 as discussed in Pinto, Koay et al. 2018
def bin_inputs(unnnorm_inpt, subset_bins=True):
    if subset_bins == True:
        bins = np.arange(-6, 6, 0.5)
    else:
        bins = np.arange(min(unnnorm_inpt), max(unnnorm_inpt), (max(unnnorm_inpt)-min(unnnorm_inpt))/8)
    click_bins = np.digitize(unnnorm_inpt, bins=bins)
    bin_diff_dict = {}
    for q, this_bin in enumerate(np.unique(click_bins)):
        mean_diff = np.mean(unnnorm_inpt[click_bins == this_bin])
        bin_diff_dict[this_bin] = mean_diff#bins[q]
    # Now create a column to replace diff with binned_diff
    binned_click_diff = np.array(pd.Series(click_bins).map(bin_diff_dict))
    return binned_click_diff

def read_psychometric_from_file(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    stim_vals = data[0]
    psychometric = data[1]
    return stim_vals, psychometric

if __name__ == '__main__':
    data_dir = 'data/human/data_for_cluster/data_by_animal/'
    figure_dir = 'figures/figures_for_paper/figure_8/'
    animal_list = animal_list = load_animal_list(data_dir + 'animal_list.npz')

    cols = ['#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

    covar_set = 2
    alpha_val = 2.0
    sigma_val = 0.5
    K = 2
    D, M, C = 1, 3, 2
    include_psychometric = False

    global_weights, _ = get_global_weights(K)
    fig = plt.figure(figsize=(5, 3.3))
    plt.subplots_adjust(wspace=0.5, hspace=0.8)
    for k in range(K):
        plt.subplot(2, 3, k+1)

        for animal in animal_list:
            results_dir = 'results/human_individual_fit_prior_2/' + animal + '/'

            cv_file = results_dir + "/cvbt_folds_model.npz"
            cvbt_folds_model = load_cv_arr(cv_file)


            with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
                best_init_cvbt_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)

            transition_matrix = np.exp(hmm_params[1][0])
            weight_vectors = -hmm_params[2]

            plt.plot(range(M + 1), weight_vectors[k][0][[0, 3, 1, 2]], '-o', color=cols[k],
                     lw=1, alpha = 0.7, markersize = 3)
        if k ==0:
            plt.yticks([-2, 0, 2, 4], fontsize=10)
            plt.xticks([0, 1, 2, 3], ['stim.', 'bias', 'p.c.', 'w.s.l.s.'], fontsize = 10, rotation = 20)
            plt.ylabel("GLM weight", fontsize = 10)
        else:
            plt.yticks([-2, 0, 2, 4], ['', '','',''])
            plt.xticks([0, 1, 2, 3], ['', '','',''])
        plt.title("state " + str(k+1), fontsize = 10, color= cols[k])
        #legend = plt.legend(title = "State", fontsize=10,  ncol = 3, columnspacing = 0.1, loc = 'upper right')
        plt.plot(range(M + 1), global_weights[k][0][[0, 3, 1, 2]], '-o', color='k',
                 lw=1.3, alpha=1, markersize=3, label = 'global')
        plt.axhline(y=0, color="k", alpha=0.5, ls="--", linewidth = 0.75)
        if k ==1:
            plt.legend(fontsize=10, labelspacing=0.2, handlelength=1, borderaxespad=0.2, borderpad=0.2, framealpha=0,
                       bbox_to_anchor=(0.53, 1), handletextpad=0.2, loc='upper left')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(0, 0, 1, 1)
        plt.ylim((-2, 4.9))

    # ==================================== PSYCHOMETRICS ==================================
    data_dir = 'data/human/data_for_cluster/'
    K = 2
    covar_set = 2
    inpt, old_y, session = load_data(data_dir + 'all_animals_concat.npz')
    unnormalized_inpt, _, _ = load_data(data_dir + 'all_animals_concat_unnormalized.npz')
    y = np.copy(old_y)

    # Restrict to non-violation trials:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, train_masks = partition_data_by_session(inpt, y, mask, session)

    # Save parameters for initializing individual fits
    weight_vectors, hmm_params = get_global_weights(K)

    # Also get data for animal:
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K), 1, 100)
    _, counts = np.unique(np.argmax(posterior_probs, axis = 1), return_counts=True)

    cols = ['#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    labels = ["'biased left'", "'biased right'", "'win stay'"]
    plt.subplot(2, 3, 3)
    # state-specific psychometrics
    for k in range(K):
        # Get index of trials where posterior_probs > 0.9 and not violation
        idx_of_interest = np.where((posterior_probs[:, k] >= 0.9) & (mask == 1))[0]
        inpt_this_state, unnormalized_inpt_this_state, y_this_state = inpt[idx_of_interest, :], unnormalized_inpt[
                                                                                                idx_of_interest,
                                                                                                :], old_y[
                                                                                                    idx_of_interest, :]
        # Fit psychometric
        uniq_contrast_vals = np.unique(unnormalized_inpt_this_state[:, 0])

        # Get accuracy
        not_zero_loc = np.where(unnormalized_inpt_this_state[:, 0] != 0)[0]
        correct_ans = (np.sign(unnormalized_inpt_this_state[not_zero_loc, 0]) + 1) / 2
        acc = np.sum(y_this_state[not_zero_loc, 0] == correct_ans) / len(correct_ans)
        # Calculate psychometric
        inpt_df = pd.DataFrame({'signed_contrast': unnormalized_inpt_this_state[:, 0], 'choice': y_this_state[:, 0]})
        psy_stats_df = calculate_psychometric_stats(unnormalized_inpt_this_state[:, 0], y_this_state[:, 0],
                                                    uniq_contrast_vals)
        stim_vals, prob_right_max = get_prob_right(weight_vectors, inpt, k, 1, 1)
        _, prob_right_min = get_prob_right(weight_vectors, inpt, k, -1, -1)

        plt.plot(stim_vals, prob_right_max, '-', color=cols[k], alpha=1, lw=1, zorder=5)  # went R and was rewarded
        plt.plot(stim_vals, get_prob_right(weight_vectors, inpt, k, -1, 1)[1], '--', color=cols[k], alpha=0.5,
                 lw=1)  # went L and was not rewarded
        plt.plot(stim_vals, get_prob_right(weight_vectors, inpt, k, 1, -1)[1], '-', color=cols[k], alpha=0.5, lw=1,
                 markersize=3)  # went R and was not rewarded
        plt.plot(stim_vals, prob_right_min, '--', color=cols[k], alpha=1, lw=1)  # went L and was rewarded

        if k == 0:
            plt.ylabel("p(choice = 'more')", fontsize=10)
            plt.xlabel('sensory evidence (a.u.)', fontsize=10)
            plt.xticks([-1.8417554175746027, -0.014912144544934091, 1.8119311284847344], labels=['-5', '0', '5'],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], labels=['0', '0.5', '1'], fontsize=10)
        else:
            plt.xticks([-1.8417554175746027, -0.014912144544934091, 1.8119311284847344], labels=['', '', ''],
                       fontsize=10)
            plt.yticks([0, 0.5, 1], labels=['', '', ''], fontsize=10)
            plt.ylabel('')
            plt.xlabel('')
        plt.ylim((0, 1))
        acc = 100 * np.around(acc, decimals=2)

        plt.axhline(y=0.5, color="k", alpha=0.5, ls="--", linewidth=0.75)
        plt.axvline(x=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(0, 0, 1, 1)

    # plot classic psychometric:
    binned_inpt = bin_inputs(inpt[:, 0], subset_bins=False)

    # Fit psychometric
    uniq_contrast_vals = np.unique(binned_inpt)
    # Plot psychometric for all animals together:
    # Calculate psychometric
    inpt_df = pd.DataFrame({'signed_contrast': binned_inpt, 'choice': y[:, 0]})
    stim_vals, psychometric = read_psychometric_from_file(figure_dir + 'psychometric_grid.npz')
    plt.plot(stim_vals, np.mean(psychometric, axis=0), '-', color='k',
             linewidth=0.9)
    g = sns.lineplot(inpt_df['signed_contrast'], inpt_df['choice'], err_style="bars", linewidth=0,
                     linestyle='None', mew=0,
                     marker='o', markersize=3, ci=68, err_kws={"linewidth": 0.7}, color=(193 / 255, 39 / 255, 45 / 255))
    plt.ylabel("p(choice = 'more')", fontsize=10)
    plt.xlabel('sensory evidence (a.u.)', fontsize=10)
    plt.xticks([-1.8417554175746027, -0.014912144544934091, 1.8119311284847344], labels=['-5', '0', '-5'],
               fontsize=10)
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
    plt.ylim((-0.05, 1.05))
    plt.axhline(y=0.5, color="k", alpha=0.5, ls="--", linewidth=0.75)
    plt.axvline(x=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    # ==================================== DWELL TIMES ==================================

    data_dir = 'data/human/data_for_cluster/data_by_animal/'
    covar_set = 2
    alpha_val = 2.0
    sigma_val = 0.5
    K = 2
    D, M, C = 1, 3, 2

    global_transition_matrix = get_global_trans_mat(K)
    global_dwell_times = 1 / (np.ones(K) - global_transition_matrix.diagonal())

    state_dwell_times = np.zeros((len(animal_list), K))
    for k in range(K):
        plt.subplot(2, 3, 4+k)
        for z, animal in enumerate(animal_list):
            results_dir = 'results/human_individual_fit_prior_2/' + animal + '/'

            cv_file = results_dir + "/cvbt_folds_model.npz"
            cvbt_folds_model = load_cv_arr(cv_file)

            with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
                best_init_cvbt_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)

            transition_matrix = np.exp(hmm_params[1][0])

            state_dwell_times[z, :] = 1 / (np.ones(K) - transition_matrix.diagonal())

        logbins = np.logspace(np.log10(1), np.log10(max(state_dwell_times[:, k])), 15)
        plt.hist(state_dwell_times[:, k], bins=logbins, color=cols[k], histtype='bar',
                 rwidth=0.8)
        ax1 = plt.gca()
        ax1.set_xscale('log')
        ax1.set_xticks([1, 10, 100, 1000])
        ax1.set_yticks([0, 2, 4, 6])
        plt.ylim((0, 6))
        plt.axvline(np.median(state_dwell_times[:, k]), linestyle='--', color='k', lw=1, label = 'median')
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if k == 0:
            ax1.set_yticklabels(labels=["0", "2", "4", "6"], fontsize=10, alpha=1)
            ax1.set_xticklabels(labels=["1", "10", "100", "1000"], fontsize=10, alpha=1, rotation=45)
            plt.ylabel("# participants", fontsize = 10)
            plt.xlabel("expected dwell time \n (# trials)", fontsize = 10, labelpad=0)
        else:
            ax1.set_yticklabels(labels=["", "", "", ""], fontsize=10, alpha=1)
            ax1.set_xticklabels(labels=["", "", "", ""], fontsize=10, alpha=1, rotation=45)
        if k ==1:
            plt.legend(fontsize=10, labelspacing=0.2, handlelength=0.8, borderaxespad=0.2, borderpad=0.2, framealpha=0,
                       bbox_to_anchor=(-0.01, 1.12), handletextpad=0.2, loc='upper left')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(0, 0, 1, 1)


    # =================== NUM SWITCHES FIGURE =====================
    cp_hist, cp_bin_locs, num_sess_all_animals, change_points_per_sess = load_switches_data(figure_dir + 'num_switches_data.npz')
    plt.subplot(2, 3, 6)
    frac_non_zero = 0
    for z, occ in enumerate(cp_hist):
        plt.bar(cp_bin_locs[z], occ / num_sess_all_animals, width=2, color=(66 / 255, 150 / 255, 129 / 255))
        if cp_bin_locs[z] > 0:
            frac_non_zero += occ / num_sess_all_animals
    # Show median across sessions:
    plt.axvline(x=np.median(change_points_per_sess), color='k', label = 'median')
    plt.xticks([0, 10, 20, 30], fontsize=10)
    plt.yticks([0, 0.1, 0.2], fontsize=10)
    plt.xlabel('# state changes \n'
               'in session', fontsize=10)
    plt.ylabel('frac. sessions', fontsize=10)  # , labelpad=-0.5)
    # plt.title('incl. first 100 trials', fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.legend(fontsize=10, labelspacing=0.2, handlelength=1, borderaxespad=0.2, borderpad=0.2, framealpha=0,
               bbox_to_anchor=(0.5, 1), handletextpad=0.2, loc='upper left')


    fig.savefig(figure_dir + 'massive_grid.pdf')
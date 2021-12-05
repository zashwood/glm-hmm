# Save best parameters (global fit, Odoemene) for initializing individual fits
import numpy as np
import json
import pandas as pd
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, calculate_state_permutation, partition_data_by_session, create_violation_mask, get_marginal_posterior
from io_utils import load_cv_arr, load_glmhmm_data, load_data, load_session_fold_lookup, load_animal_list
import matplotlib.pyplot as plt
import seaborn as sns
from psychofit import mle_fit_psycho, sigmoid_psycho_2gammas
#import ssm
from compHess import compHess
from scipy.special import logsumexp

#import psytrack as psy
import matplotlib
from scipy.special import expit, logit


matplotlib.font_manager._rebuild()
#colors = psy.COLORS
#zorder = psy.ZORDER
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
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

def get_global_weights(covar_set):
    results_dir = 'results/odoemene_global_fit/' + 'covar_set_' + str(
        covar_set)

    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    global_weights = -hmm_params[2]
    return global_weights

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


def get_global_trans_mat(covar_set):
    results_dir = 'results/odoemene_global_fit/' + 'covar_set_' + str(
        covar_set)

    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    global_weights = -hmm_params[2]
    global_transition_matrix = np.exp(hmm_params[1][0])
    return global_transition_matrix

def get_prob_right(weight_vectors, inpt, k, pc, wsls):
    # stim vector
    min_val_stim = np.min(inpt[:,0])
    max_val_stim = np.max(inpt[:,0])
    stim_vals = np.arange(min_val_stim, max_val_stim, 0.05)
    # create input matrix - cols are stim, pc, wsls, bias
    x = np.array([stim_vals, np.repeat(pc, len(stim_vals)), np.repeat(wsls, len(stim_vals)), np.repeat(1, len(stim_vals))]).T
    wx = np.matmul(x, weight_vectors[k][0])
    return stim_vals, expit(wx)

if __name__ == '__main__':
    data_dir = 'data/odoemene/data_for_cluster/data_by_animal/'
    figure_dir = 'figures/figures_for_paper/figure_5/'
    animal_list = animal_list = load_animal_list(data_dir + 'animal_list.npz')
    #animal = 'CSHL_008'

    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

    covar_set = 2
    alpha_val = 2.0
    sigma_val = 0.5
    K = 4
    D, M, C = 1, 3, 2
    include_psychometric = False

    global_weights = get_global_weights(covar_set)
    fig = plt.figure(figsize=(5.4, 4.5))
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    for k in range(K):
        plt.subplot(3, 4, k+1)

        for animal in animal_list:
            results_dir = 'results/odoemene_individual_fit/' + 'covar_set_' + str(
                covar_set) + '/prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'


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
        plt.plot(range(M + 1), global_weights[k][0][[0, 3, 1, 2]], '-o', color='k',
                 lw=1.3, alpha=1, markersize=3, label = 'global')
        plt.axhline(y=0, color="k", alpha=0.5, ls="--", linewidth = 0.75)
        if k ==1:
            plt.legend(fontsize=10, labelspacing=0.2, handlelength=1, borderaxespad=0.2, borderpad=0.2, framealpha=0,
                       bbox_to_anchor=(0.53, 1), handletextpad=0.2, loc='upper left')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(0, 0, 1, 1)
        plt.ylim((-2, 4))

    # ==================================== PSYCHOMETRICS ==================================
    data_dir = 'data/odoemene/data_for_cluster/'
    K = 4
    covar_set = 2
    inpt, old_y, session = load_data(data_dir + 'all_animals_concat.npz')
    print(np.unique(inpt[:,0]))
    unnormalized_inpt, _, _ = load_data(data_dir + 'all_animals_concat_unnormalized.npz')
    y = np.copy(old_y)

    # Restrict to non-violation trials:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    y[np.where(y == -1), :] = 1
    inputs, datas, train_masks = partition_data_by_session(inpt, y, mask, session)

    covar_set = 2
    # Get posterior probs:
    results_dir = 'results/odoemene_global_fit/' + 'covar_set_' + str(
        covar_set)

    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)

    # Save parameters for initializing individual fits
    weight_vectors = hmm_params[2]
    log_transition_matrix = hmm_params[1][0]
    init_state_dist = hmm_params[0][0]

    # Also get data for animal:

    posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K), 1, 100)
    _, counts = np.unique(np.argmax(posterior_probs, axis = 1), return_counts=True)

    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    labels = ["'engaged'", "'biased left'", "'biased right'", "'win stay'"]
    for k in range(K):
        # Get index of trials where posterior_probs > 0.9 and not violation
        idx_of_interest = np.where((posterior_probs[:, k] >= 0.9) & (mask == 1))[0]
        inpt_this_state, unnormalized_inpt_this_state, y_this_state = inpt[idx_of_interest, :], unnormalized_inpt[
                                                                                                idx_of_interest,
                                                                                                :], old_y[
                                                                                                    idx_of_interest, :]
        # Fit psychometric
        uniq_contrast_vals = np.unique(unnormalized_inpt_this_state[:, 0])
        # Plot psychometric for all animals together:
        plt.subplot(3, 4, k+5)
        # Get accuracy
        not_zero_loc = np.where(unnormalized_inpt_this_state[:, 0] != 0)[0]
        correct_ans = (np.sign(unnormalized_inpt_this_state[not_zero_loc, 0]) + 1) / 2
        acc = np.sum(y_this_state[not_zero_loc, 0] == correct_ans) / len(correct_ans)
        # Calculate psychometric
        inpt_df = pd.DataFrame({'signed_contrast': unnormalized_inpt_this_state[:, 0], 'choice': y_this_state[:, 0]})
        psy_stats_df = calculate_psychometric_stats(unnormalized_inpt_this_state[:, 0], y_this_state[:, 0],
                                                    uniq_contrast_vals)
        if include_psychometric == True:
            pars, L = mle_fit_psycho(psy_stats_df.transpose().values,  # extract the data from the df
                                     P_model='sigmoid_psycho_2gammas',
                                     parstart=np.array([psy_stats_df['signed_contrast'].mean(), 20., 0.05, 0.05]),
                                     parmin=np.array([psy_stats_df['signed_contrast'].min(), -300, 0., 0.]),
                                     parmax=np.array([psy_stats_df['signed_contrast'].max(), 300., 0.5, 0.5]),
                                     nfits=100)
            plt.plot(np.arange(-8.5, 8.5, 0.5), sigmoid_psycho_2gammas(pars, np.arange(-8.5, 8.5, 0.5)), '-', color=cols[k])
            g = sns.lineplot(inpt_df['signed_contrast'], inpt_df['choice'], color=cols[k], err_style="bars", linewidth=0,
                             linestyle='None', mew=0,
                             marker='o', markersize=2, ci=68)
            if k == 0:
                plt.ylabel('p("R")', fontsize=10)
                plt.xlabel('flash rate (Hz)', fontsize=10)
                plt.xticks([-8, 0, 8], labels=['4', '12', '20'],
                           fontsize=10)
                plt.yticks([0, 0.5, 1], labels=['0', '0.5', '1'], fontsize=10)
            else:
                plt.xticks([-8, 0, 8], labels=['', '', ''],
                           fontsize=10)
                plt.yticks([0, 0.5, 1], labels=['', '', ''], fontsize=10)
                plt.ylabel('')
                plt.xlabel('')
            g.set_ylim([0, 1])
        else:
            # USE GLM WEIGHTS TO GET PROB RIGHT
            stim_vals, prob_right_max = get_prob_right(-weight_vectors, inpt, k, 1, 1)
            _, prob_right_min = get_prob_right(-weight_vectors, inpt, k, -1, -1)

            plt.plot(stim_vals, prob_right_max, '-', color=cols[k], alpha=1, lw=1, zorder=5)  # went R and was rewarded
            plt.plot(stim_vals, get_prob_right(-weight_vectors, inpt, k, -1, 1)[1], '--', color=cols[k], alpha=0.5,
                     lw=1)
            plt.plot(stim_vals, get_prob_right(-weight_vectors, inpt, k, 1, -1)[1], '-', color=cols[k], alpha=0.5, lw=1,
                     markersize=3)
            plt.plot(stim_vals, prob_right_min, '--', color=cols[k], alpha=1, lw=1)  # went L and was rewarded

            if k == 0:
                plt.ylabel('p("R")', fontsize=10)
                plt.xlabel('flash rate (Hz)', fontsize=10)
                plt.xticks([-1.26321895, -0.00217184, 1.25887527], labels=['4', '12', '20'],
                           fontsize=10)
                plt.yticks([0, 0.5, 1], labels=['0', '0.5', '1'], fontsize=10)
            else:
                plt.xticks([-1.26321895, -0.00217184, 1.25887527], labels=['', '', ''],
                           fontsize=10)
                plt.yticks([0, 0.5, 1], labels=['', '', ''], fontsize=10)
                plt.ylabel('')
                plt.xlabel('')
            plt.ylim((0, 1))
        acc = 100 * np.around(acc, decimals=2)

        plt.title(labels[k], fontsize = 10, color = cols[k])

        plt.axhline(y=0.5, color="k", alpha=0.5, ls="--", linewidth=0.75)
        plt.axvline(x=0, color="k", alpha=0.5, ls="--", linewidth=0.75)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(0, 0, 1, 1)

    # ==================================== DWELL TIMES ==================================

    data_dir = 'data/odoemene/data_for_cluster/data_by_animal/'
    covar_set = 2
    alpha_val = 2.0
    sigma_val = 0.5
    K = 4
    D, M, C = 1, 3, 2

    global_transition_matrix = get_global_trans_mat(covar_set)
    global_dwell_times = 1 / (np.ones(K) - global_transition_matrix.diagonal())

    state_dwell_times = np.zeros((len(animal_list), K))
    for k in range(K):
        plt.subplot(3,4, k+9)
        for z, animal in enumerate(animal_list):
            results_dir = 'results/odoemene_individual_fit/' + 'covar_set_' + str(
                covar_set) + '/prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(
                alpha_val) + '/' + animal + '/'

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
        # plt.xlim((0, 500))
        ax1 = plt.gca()
        ax1.set_xscale('log')
        ax1.set_xticks([1, 10, 100, 1000])
        ax1.set_yticks([0, 2, 4])
        plt.ylim((0, 4))
        plt.axvline(np.median(state_dwell_times[:, k]), linestyle='--', color='k', lw=1, label = 'median')
        #plt.axvline(global_dwell_times[k], linestyle='-', color='k', lw=1, label='global')
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if k == 0:
            ax1.set_yticklabels(labels=["0", "2", "4"], fontsize=10, alpha=1)
            ax1.set_xticklabels(labels=["1", "10", "100", "1000"], fontsize=10, alpha=1, rotation=45)
            plt.ylabel("# animals", fontsize = 10)
            plt.xlabel("expected dwell time \n (# trials)", fontsize = 10, labelpad=0)
        else:
            ax1.set_yticklabels(labels=["", "", ""], fontsize=10, alpha=1)
            ax1.set_xticklabels(labels=["", "", "", ""], fontsize=10, alpha=1, rotation=45)
        if k ==0:
            plt.legend(fontsize=10, labelspacing=0.2, handlelength=0.8, borderaxespad=0.2, borderpad=0.2, framealpha=0,
                       bbox_to_anchor=(-0.03, 1), handletextpad=0.2, loc='upper left')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(0, 0, 1, 1)

    fig.savefig(figure_dir + 'figure_5.pdf')


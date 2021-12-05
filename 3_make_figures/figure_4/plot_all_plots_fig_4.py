# Save best parameters (global fit, Odoemene) for initializing individual fits
import numpy as np
import json
import pandas as pd
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, calculate_state_permutation, partition_data_by_session, create_violation_mask, get_marginal_posterior
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

def get_global_weights(covar_set):
    results_dir = 'results/ibl_global_fit/' + 'covar_set_' + str(
        covar_set)

    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    permutation = calculate_state_permutation(hmm_params)
    global_weights = -hmm_params[2][permutation]
    return global_weights

def get_global_trans_mat(covar_set):
    results_dir = 'results/ibl_global_fit/' + 'covar_set_' + str(
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


def load_correct_incorrect_mat(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    correct_mat = data[0]
    num_trials = data[1]
    return correct_mat, num_trials

if __name__ == '__main__':
    data_dir = 'data/ibl/data_for_cluster/data_by_animal/'
    figure_dir = 'figures/figures_for_paper/figure_3/'
    animal_list = animal_list = load_animal_list(data_dir + 'animal_list.npz')

    cols = ['#4daf4a', '#ff7f00', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

    covar_set = 2
    alpha_val = 2
    sigma_val = 2
    K = 3
    D, M, C = 1, 3, 2

    global_weights = get_global_weights(covar_set)

    fig = plt.figure(figsize=(5.5, 5.5))
    plt.subplots_adjust(wspace=0.45, hspace=0.6)

    # =========== NLL ====================================
    plt.subplot(3, 3, 1)
    for animal in animal_list:
        covar_set = 2
        alpha_val = 2
        sigma_val = 2
        K = 3
        D, M, C = 1, 3, 2

        results_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
            covar_set) + '/prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'
        cols = ['#999999', '#984ea3', '#e41a1c', '#dede00']
        cv_arr = load_cv_arr(results_dir + "/cvbt_folds_model.npz")

        cv_arr_for_plotting = cv_arr[[0, 2, 3, 4, 5, 6], :]
        mean_cvbt = np.mean(cv_arr_for_plotting, axis=1)
        if animal == "CSHL_008":
            plt.plot([0, 0.5, 1, 2, 3, 4], mean_cvbt - mean_cvbt[0], '-o', color = 'k', linestyle='--', zorder = 2, alpha = 0.7, lw = 2, markersize = 4, label = 'example mouse')
        else:
            plt.plot([0, 0.5, 1, 2, 3, 4], mean_cvbt - mean_cvbt[0], '-o', color=cols[0], zorder=0, alpha=0.6, lw=1.5,
                     markersize=4)
    # Get global
    results_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
        covar_set)+ '/prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/'
    cv_arr = load_cv_arr(results_dir + "/cvbt_folds_model.npz")
    cv_arr_for_plotting = cv_arr[[0, 2, 3, 4, 5, 6], :]
    mean_cvbt = np.mean(cv_arr_for_plotting, axis=1)
    plt.plot([0, 0.5, 1, 2, 3, 4], mean_cvbt - mean_cvbt[0], '-o', color='k', zorder=1, alpha=1, lw=1.5, markersize = 4, label = 'mean')
    plt.xticks([0, 0.5, 1, 2, 3, 4], ['1', 'L.', '2', '3', '4', '5'], fontsize = 10)
    plt.ylabel("$\Delta$ test LL (bits/trial)", fontsize=10, labelpad=0 )
    plt.xlabel("# states", fontsize=10, labelpad=0)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.ylim((-0.01, 0.24))
    #plt.yticks(color = cols[0])
    leg = plt.legend(fontsize=10, labelspacing=0.05, handlelength=1.4, borderaxespad=0.05, borderpad=0.05, framealpha=0,
               bbox_to_anchor=(1.00, 0.9), loc='lower right', markerscale=0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(1.0)


    # =========== PRED ACC =========================
    plt.subplot(3, 3, 2)
    mean_across_animals = []
    num_trials_all_animals = 0
    min = 10
    max = 0
    for z, animal in enumerate(animal_list):
        covar_set = 2
        alpha_val = 2
        sigma_val = 2
        K = 3
        D, M, C = 1, 3, 2

        results_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
            covar_set) + '/prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'

        cols = ['#999999', '#984ea3' , '#e41a1c', '#dede00']

        correct_mat, num_trials = load_correct_incorrect_mat(results_dir + "correct_incorrect_mat.npz")
        num_trials_all_animals += np.sum(num_trials)
        if z == 0:
            trials_correctly_predicted_all_folds = np.sum(correct_mat, axis=1)
        else:
            trials_correctly_predicted_all_folds = trials_correctly_predicted_all_folds + np.sum(correct_mat, axis=1)

        pred_acc_arr = load_cv_arr(results_dir + "predictive_accuracy_mat.npz")
        pred_acc_arr_for_plotting = pred_acc_arr[[0, 2, 3, 4, 5, 6], :]

        mean_acc = np.mean(pred_acc_arr_for_plotting, axis=1)
        if animal == "CSHL_008":
            plt.plot([0, 0.5, 1, 2, 3, 4], mean_acc - mean_acc[0], '-o', color='k', linestyle='--', zorder=2, alpha=0.7,
                     lw=2, markersize=4, label='example mouse')
        else:
            plt.plot([0, 0.5, 1, 2, 3, 4], mean_acc - mean_acc[0], '-o', color=cols[1], zorder=0, alpha=0.6,
                     lw=1.5, markersize=4)
        mean_across_animals.append(mean_acc - mean_acc[0])
    ymin = -0.01
    ymax = 0.15
    plt.xticks([0, 0.5, 1, 2, 3, 4], ['1', 'L.', '2', '3', '4', '5'], fontsize = 10)
    plt.ylim((ymin, ymax))
    trial_nums = (trials_correctly_predicted_all_folds - trials_correctly_predicted_all_folds[0])
    plt.plot([0, 0.5, 1, 2, 3, 4], np.mean(mean_across_animals, axis=0), '-o', color='k', zorder=0, alpha=1,
             markersize=4, lw=1.5, label='mean')
    plt.xlabel("# states", fontsize=10, labelpad=0)
    plt.yticks([0, 0.05, 0.1], ["0", "5%", '10%'])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)

    # ================ SIMPLEX =======================
    plt.subplot(3, 3, 3)
    animal_list = load_animal_list(data_dir + 'animal_list.npz')
    overall_frac_state_1 = []
    overall_frac_state_2 = []
    overall_frac_state_3 = []
    frac_state_1 = []
    frac_state_2 = []
    frac_state_3 = []
    num_sessions_single_state = 0
    num_sessions = 0
    for animal in animal_list:
        results_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
            covar_set) + '/' + 'prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(
            alpha_val) + '/' + animal + '/'
        figure_dir = 'figures/figures_for_paper/figure_3/'

        cv_file = results_dir + "/cvbt_folds_model.npz"
        cvbt_folds_model = load_cv_arr(cv_file)

        K = 3
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
        inpt, y, session = load_data(data_dir + animal + '_processed.npz')

        # Create mask:
        # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
        y[np.where(y == -1), :] = 1
        inputs, datas, train_masks = partition_data_by_session(inpt, y, mask, session)

        if animal != "ibl_witten_05" and animal != "CSHL_001":
            posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K), alpha_val,
                                                     sigma_val)
        else:
            permutation = calculate_state_permutation(hmm_params)
            posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, permutation, alpha_val,
                                                     sigma_val)

        # Get overall state occupancies
        max_posterior_probs = np.argmax(posterior_probs, axis=1)
        vals, counts = np.unique(max_posterior_probs, return_counts=True)
        if 0 in vals:
            idx = np.where(vals == 0)
            overall_frac_state_1.append(counts[idx] / np.sum(counts))
        else:
            overall_frac_state_1.append(0)
        if 1 in vals:
            idx = np.where(vals == 1)
            overall_frac_state_2.append(counts[idx] / np.sum(counts))
        else:
            overall_frac_state_2.append(0)
        if 2 in vals:
            idx = np.where(vals == 2)
            overall_frac_state_3.append(counts[idx] / np.sum(counts))
        else:
            overall_frac_state_3.append(0)

        sess_to_plot = np.unique(session)

        cols = ['#984ea3', '#999999', '#e41a1c', '#dede00']
        for i, sess in enumerate(sess_to_plot):
            idx_session = np.where(session == sess)
            this_inpt = inpt[idx_session[0], :]
            max_posterior_probs_this_session = np.argmax(posterior_probs[idx_session[0], :], axis=1)
            vals, counts = np.unique(max_posterior_probs_this_session, return_counts=True)
            num_sessions += 1
            if counts[0] == len(max_posterior_probs_this_session):
                num_sessions_single_state += 1

            if 0 in vals:
                idx = np.where(vals == 0)
                frac_state_1.append(counts[idx] / np.sum(counts))
            else:
                frac_state_1.append(0)

            if 1 in vals:
                idx = np.where(vals == 1)
                frac_state_2.append(counts[idx] / np.sum(counts))
            else:
                frac_state_2.append(0)

            if 2 in vals:
                idx = np.where(vals == 2)
                frac_state_3.append(counts[idx] / np.sum(counts))
            else:
                frac_state_3.append(0)

    plt.scatter(frac_state_1, frac_state_2, alpha=0.2, color=cols[1], s=1, label='session')
    plt.scatter(overall_frac_state_1, overall_frac_state_2, alpha=0.3, color='r', s=2, label='mouse', zorder = 2)
    overall_frac_state_1 = np.array(overall_frac_state_1)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.ylabel("frac. state 2", fontsize=10, labelpad=0)
    plt.xlabel("frac. state 1", fontsize=10,  labelpad=0)
    plt.xticks([0, 0.5, 1], ["0", "0.5", "1"])
    plt.yticks([0, 0.5, 1], ["0", "0.5", "1"])
    plt.legend(fontsize=10, labelspacing=0.1, handlelength=0.3, borderaxespad=0.1, borderpad=0.1, framealpha=0,
               bbox_to_anchor=(1, 1), loc='upper right', markerscale=2.5)


    # ==================== WEIGHTS =======================
    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    for k in range(K):
        plt.subplot(3, 3, k+4)
        for animal in animal_list:
                results_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
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

                if animal == "ibl_witten_05" or animal == "CSHL_001":
                    permutation = calculate_state_permutation(hmm_params)
                    weight_vectors = weight_vectors[permutation]
                    transition_matrix = permute_transition_matrix(transition_matrix, permutation)

                if animal=="CSHL_008":
                    plt.plot(range(M + 1), global_weights[k][0][[0, 3, 1, 2]], '-o', color='k',
                             lw=1.3, alpha=1, markersize=3, zorder=1, label='global fit')
                    plt.plot(range(M + 1), weight_vectors[k][0][[0, 3, 1, 2]], '-o', color='k',linestyle = '--',
                             lw=1, alpha = 0.7, markersize = 3, zorder = 2, label = 'example mouse')
                else:
                    plt.plot(range(M + 1), weight_vectors[k][0][[0, 3, 1, 2]], '-o', color=cols[k],
                             lw=1, alpha = 0.7, markersize = 3, zorder = 0)

        if k ==0:
            plt.yticks([-3, 0, 3, 6, 9, 12], fontsize=10)
            plt.xticks([0, 1, 2, 3], ['stim.', 'bias', 'p.c.', 'w.s.l.s.'], fontsize = 10, rotation = 45)
            plt.ylabel('GLM weight', fontsize = 10)
        else:
            plt.yticks([-3, 0, 3, 6, 9, 12], ['', '','','', '', ''])
            plt.xticks([0, 1, 2, 3], ['', '','',''])

        plt.axhline(y=0, color="k", alpha=0.5, ls="--", linewidth = 0.75)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(0, 0, 1, 1)
        plt.ylim((-4, 14))
        if k ==0:
            plt.legend(fontsize = 10,  labelspacing=0.2, handlelength =1.4, borderaxespad = 0.2, borderpad=0.2, framealpha = 0, bbox_to_anchor=(0.3, 0.8))

    # ==================== TRANSITIONS ========================
    global_transition_matrix = get_global_trans_mat(covar_set)
    global_dwell_times = 1 / (np.ones(K) - global_transition_matrix.diagonal())

    state_dwell_times = np.zeros((len(animal_list), K))
    example_animal_vals = np.zeros(K)
    for k in range(K):
        plt.subplot(3, 3, k + 7)
        for z, animal in enumerate(animal_list):
            results_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
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

            if animal == "ibl_witten_05" or animal == "CSHL_001":
                permutation = calculate_state_permutation(hmm_params)
                transition_matrix = permute_transition_matrix(transition_matrix, permutation)

            state_dwell_times[z, :] = 1 / (np.ones(K) - transition_matrix.diagonal())
            if animal == "CSHL_008":
                example_animal_vals = 1 / (np.ones(K) - transition_matrix.diagonal())

        plt.hist(state_dwell_times[:, k], bins=np.arange(0, 90, 5), color=cols[k], histtype='bar',
                 rwidth=0.8)
        plt.ylim(0, 12)
        if k ==0:
            plt.xticks([0, 20, 40, 60, 80], ["0", "", "40", "", "80"], fontsize = 10)
            plt.yticks([0, 5, 10], fontsize = 10)
            plt.ylabel("# mice", fontsize = 10)
            plt.xlabel("expected dwell time \n (# trials)", fontsize = 10)
        else:
            plt.xticks([0, 20, 40, 60, 80], ["", "", "", "", ""], fontsize=10)
            plt.yticks([0, 5, 10], ["", "", ""], fontsize=10)
        plt.axvline(np.median(state_dwell_times[:, k]), linestyle='-', color='k', lw=1, label = 'median')
        plt.axvline(example_animal_vals[k], linestyle='--', color='k', alpha=0.7, lw=1, label = 'ex. animal')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.subplots_adjust(0, 0, 1, 1)

    fig.savefig(figure_dir + 'params_all_animals.pdf')

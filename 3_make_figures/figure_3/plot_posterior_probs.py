import numpy as np
import json
from io_utils import load_cv_arr, load_glmhmm_data, load_data, load_session_fold_lookup, load_animal_list
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, calculate_state_permutation, get_marginal_posterior, partition_data_by_session, create_violation_mask
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

def load_glm_hmm_data(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    posterior_probs = data[0]
    inpt = data[1]
    y = data[2]
    session = data[3]
    weight_vectors = data[4]
    transition_matrix = data[5]
    init_state_dist = data[6]
    return posterior_probs, inpt, y, session, weight_vectors, transition_matrix, init_state_dist

def get_was_correct(this_inpt, this_y):
    '''
    return a vector of size this_y.shape[0] indicating if
    choice was correct on current trial.  Return NA if trial was not "easy" trial
    :param this_inpt:
    :param this_y:
    :return:
    '''
    was_correct = np.empty(this_y.shape[0])
    was_correct [:] = np.NaN
    idx_easy= np.where(np.abs(this_inpt[:,0])>0.002)
    correct_side = (np.sign(this_inpt[idx_easy, 0])+1)/2
    was_correct[idx_easy] = (correct_side == this_y[idx_easy, 0])+0 # TODO: fix this since sign in {-1, 1}
    return was_correct, idx_easy



def get_convolved_accuracy(this_inpt, this_y):
    '''
    Given inpt and response for session, get convolved accuracy on "easy" trials for this animal
    :param this_inpt: inpt for particular session
    :param this_y: choices for particular session
    :return: average accuracy across sets of 8 trials
    '''
    import pandas as pd
    was_correct, idx_easy_trials = get_was_correct(this_inpt, this_y)
    df = pd.DataFrame({'was_correct':was_correct})
    convolved_acc = df.rolling(window=5, min_periods=1).mean()
    return convolved_acc, was_correct, idx_easy_trials



if __name__ == '__main__':
    data_dir = 'data/ibl/data_for_cluster/data_by_animal/'
    sigma_val = 2
    alpha_val = 2
    covar_set = 2
    animal = "CSHL_008"
    results_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
        covar_set) + '/' + 'prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/' + animal + '/'
    figure_dir = 'figures/figures_for_paper/figure_3/'


    np.random.seed(59)

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
    all_sessions = np.unique(session)
    if covar_set == 0:
        inpt = inpt[:, [0]]
        print(inpt.shape)
    elif covar_set == 1:
        inpt = inpt[:, [0, 1]]
        print(inpt.shape)
    elif covar_set == 2:
        inpt = inpt[:, [0, 1, 2]]
    # Create mask:
    # Identify violations for exclusion:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
    y[np.where(y==-1),:] = 1
    inputs, datas, train_masks = partition_data_by_session(inpt, y, mask, session)

    #print(hmm_params[2])
    posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K), alpha_val, sigma_val)
    states_max_posterior = np.argmax(posterior_probs, axis = 1)


    sess_to_plot = ["CSHL_008-2019-04-29-001", "CSHL_008-2019-08-07-001", "CSHL_008-2019-05-28-001"]  # [idx.astype('int')]

    cols = ['#ff7f00', '#4daf4a', '#377eb8', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    fig = plt.figure(figsize=(5.3, 4))
    plt.subplots_adjust(wspace=0.2, hspace=0.9)
    for i, sess in enumerate(sess_to_plot):
        plt.subplot(3, 3, i+4)
        idx_session = np.where(session == sess)
        this_inpt, this_y = inpt[idx_session[0], :], y[idx_session[0], :]
        was_correct, idx_easy = get_was_correct(this_inpt, this_y)
        this_y = this_y[:,0] + np.random.normal(0,0.03,len(this_y[:,0]))
        # plot choice, color by correct/incorrect:
        locs_correct = np.where(was_correct==1)[0]
        locs_incorrect = np.where(was_correct==0)[0]
        plt.plot(locs_correct, this_y[locs_correct], 'o', color = 'black', markersize = 2, zorder = 3, alpha = 0.5)
        plt.plot(locs_incorrect, this_y[locs_incorrect], 'v', color='red', markersize=2, zorder=4, alpha=0.5)

        states_this_sess = states_max_posterior[idx_session[0]]
        state_change_locs = np.where(np.abs(np.diff(states_this_sess))>0)[0]
        for change_loc in state_change_locs:
            plt.axvline(x=change_loc, color='k', lw=0.5, linestyle = '--')
        #plt.plot(range(-2, 88, 2), np.array(convolved_acc)[range(0, 90, 2)], zorder = 0, color = 'k', alpha = 0.5, lw = 1)
        plt.ylim((-0.13, 1.13))
        if i == 0:
            plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)
            plt.yticks([0, 1], ["L", "R"], fontsize=10)
        else:
            plt.xticks([0, 45, 90], ["", "", ""], fontsize=10)
            plt.yticks([0, 1], ["", ""], fontsize=10)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.title("example session " + str(i+1), fontsize = 10)
        if i ==0:
            plt.xlabel("trial #", fontsize = 10)
            plt.ylabel("choice", fontsize = 10)
        plt.subplots_adjust(0, 0, 1, 1)


    for i, sess in enumerate(sess_to_plot):
        plt.subplot(3, 3, i+1)
        idx_session = np.where(session == sess)
        this_inpt = inpt[idx_session[0], :]
        posterior_probs_this_session = posterior_probs[idx_session[0], :]
        # Plot trial structure for this session too:
        for k in range(K):
            plt.plot(posterior_probs_this_session[:, k], label="State " + str(k + 1), lw=1,
                     color=cols[k])
        states_this_sess = states_max_posterior[idx_session[0]]
        state_change_locs = np.where(np.abs(np.diff(states_this_sess)) > 0)[0]
        for change_loc in state_change_locs:
            plt.axvline(x=change_loc, color='k', lw=0.5, linestyle = '--')
        if i == 0:
            plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)
            plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=10)
        else:
            plt.xticks([0, 45, 90], ["", "", ""], fontsize=10)
            plt.yticks([0, 0.5, 1], ["", "", ""], fontsize=10)
        plt.ylim((-0.01, 1.01))
        plt.title("example session " + str(i + 1), fontsize=10)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        if i ==0:
            plt.xlabel("trial #", fontsize = 10)
            plt.ylabel("p(state)", fontsize = 10)
        plt.subplots_adjust(0, 0, 1, 1)


    # Now plot avg session:
    posterior_probs_mat = []
    for i, sess in enumerate(all_sessions):
        idx_session = np.where(session == sess)
        posterior_probs_this_session = posterior_probs[idx_session[0], :]
        if len(posterior_probs_this_session) == 90:
            posterior_probs_mat.append(posterior_probs_this_session)
    posterior_probs_mat = np.array(posterior_probs_mat)
    avg_posterior = np.mean(posterior_probs_mat, axis = 0)
    std_dev_posterior = np.std(posterior_probs_mat, axis=0)
    plt.subplot(3, 3, 7)
    for k in range(K):
        plt.plot(avg_posterior[:, k], label="State " + str(k + 1), lw=1,
                 color=cols[k])
        se = std_dev_posterior[:, k] / np.sqrt(posterior_probs_mat.shape[0])
        plt.plot(avg_posterior[:, k] + se, color=cols[k], alpha=0.2)
        plt.plot(avg_posterior[:, k] - se, color=cols[k], alpha=0.2)

    plt.xticks([0, 45, 90], ["", "", ""], fontsize=10)
    plt.yticks([0, 0.5, 1], ["", "", ""], fontsize=10)
    plt.ylim((-0.01, 1.01))
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.title("avg. session ", fontsize=10)
    plt.xlabel("trial #", fontsize=10)
    plt.ylabel("p(state)", fontsize=10)
    plt.xticks([0, 45, 90], ["0", "45", "90"], fontsize=10)
    plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=10)
    plt.show()
    fig.savefig(figure_dir + 'posterior_state_probs.pdf')


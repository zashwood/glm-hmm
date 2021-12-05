# Save best parameters (global fit, Odoemene) for initializing individual fits
import numpy as np
import json
import pandas as pd
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
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12

import matplotlib.font_manager as font_manager

matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
import matplotlib.pyplot as plt
cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"]
np.random.seed(80)

def read_rts(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    rts = data[0]
    rts_sess = data[1]
    return rts, rts_sess

def get_samples_for_pop_mean(not_viols_engaged, not_viols_disengaged, n_per_state):
    sampled_not_viols_engaged = np.random.choice(not_viols_engaged, size = n_per_state)
    sampled_not_viols_disengaged = np.random.choice(not_viols_disengaged, size=n_per_state)
    sampled_viols_engaged = n_per_state - np.sum(sampled_not_viols_engaged)
    sampled_viols_disengaged = n_per_state - np.sum(sampled_not_viols_disengaged)
    return sampled_viols_engaged, sampled_viols_disengaged
#
def read_bootstrapped_mean(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    median, lower, upper, mean_viol_rate_dist = data[0], data[1], data[2], data[3]
    return median, lower, upper, mean_viol_rate_dist

def read_bootstrapped_dist(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    bootstrap_dist = data[0]
    return bootstrap_dist

if __name__ == '__main__':
    data_dir = 'data/odoemene/data_for_cluster/data_by_animal/'
    animal_list = load_animal_list(data_dir + 'animal_list.npz')
    figure_dir = 'figures/figures_for_paper/figure_6/'

    covar_set = 2
    alpha_val = 2.0
    sigma_val = 0.5
    K = 4
    # Plot these too:
    cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"]

    weights_across_animals_df = pd.DataFrame(
        columns=['animal', 'state', 'stim_weight', 'past_choice_weight', 'wsls_weight', 'bias'])

    overall_dir = 'results/odoemene_individual_fit/' + 'covar_set_' + str(
        covar_set) +'/'+ '/prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/'

    eng_viol_all_animals = []
    dis_viol_all_animals = []
    num_eng_all_animals = []
    num_dis_all_animals = []
    diffs_vec = []
    # get mean diff across all animals:
    not_viols_engaged = []
    not_viols_disengaged = []
    n_per_state = 1000

    n_all = 0
    sampled_viols_engaged_all = 0
    sampled_viols_disengaged_all = 0

    animal_list = animal_list[[12,  1,  3 , 4, 14 , 9  ,8, 13 , 6,  5 ,10  ,7 ,2 , 0, 11]]
    fig, ax = plt.subplots(figsize =(1,2))
    plt.subplots_adjust(left=0.25, bottom=0.13, right=0.95, top=0.85, wspace=0.2, hspace=0.2)
    for z, animal in enumerate(animal_list):
        print(animal)
        results_dir =  overall_dir + animal +'/'

        cv_file = results_dir + "/cvbt_folds_model.npz"

        cvbt_folds_model = load_cv_arr(cv_file)

        K = 4
        with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
            best_init_cvbt_dict = json.load(f)

        # Get the file name corresponding to the best initialization for given K value
        raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
        hmm_params, lls = load_glmhmm_data(raw_file)

        # Save parameters for initializing individual fits
        weight_vectors = -hmm_params[2]
        log_transition_matrix = hmm_params[1][0]
        init_state_dist = hmm_params[0][0]

        # Also get data for animal:
        inpt, y, session = load_data(data_dir + animal + '_processed.npz')
        if covar_set == 0:
            inpt = inpt[:, [0]]
        elif covar_set == 1:
            inpt = inpt[:, [0, 1]]
        elif covar_set == 2:
            inpt = inpt[:, [0, 1, 2]]
        # Create mask:
        # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])
        y[np.where(y==-1),:] = 1
        inputs, datas, train_masks = partition_data_by_session(inpt, y, mask, session)

        posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, range(K), alpha_val, sigma_val)

        # Read in bootstrapped distribution for animal in question:
        bootstrap_dist = read_bootstrapped_dist(figure_dir + "violation_rates/" + animal + '_bootstrap_dist.npz')
        lower = np.quantile(np.array(bootstrap_dist), 0.025)
        median = np.quantile(np.array(bootstrap_dist), 0.5)
        upper = np.quantile(np.array(bootstrap_dist), 0.975)
        plt.plot([lower, upper], [z, z], color='k', lw=0.75, alpha =0.5)
        plt.scatter(median,z,  alpha = 0.5, color = 'k', s = 1)
        diffs_vec.append(median)

    median, lower, upper, mean_viol_rate_dist = read_bootstrapped_mean(overall_dir + 'mean_viol_rate_bootstrap.npz')
    plt.plot([lower, upper], [z+1, z+1], color = '#0343df', lw = 0.75)
    plt.scatter(median, z+1, color = 'b', s = 1)
    plt.axvline(x=0, linestyle = '--', color = 'k', alpha = 0.5, lw=0.75)
    plt.xticks([-0.05, 0, 0.05, 0.1], ["", "0", "", "10%"])
    plt.yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.ylabel("animal (Odoemene et al.)", fontsize = 10)
    plt.xlabel("$\Delta$ viol. rate \n (disengaged - \nengaged)", fontsize = 10)
    plt.show()
    fig.savefig(figure_dir + 'violation_rate_diffs.pdf')


import numpy as np
import json
from io_utils import load_cv_arr, load_glmhmm_data, load_data, load_session_fold_lookup, load_animal_list
from analyze_results_utils import get_file_name_for_best_model_fold, permute_transition_matrix, calculate_state_permutation, get_marginal_posterior, partition_data_by_session, create_violation_mask
import matplotlib.pyplot as plt

cols = ["#e74c3c", "#15b01a", "#7e1e9c", "#3498db", "#f97306"]

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
def read_rts(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    rts = data[0]
    rts_sess = data[1]
    return rts, rts_sess


def perform_bootstrap(rt_eng_vec, rt_dis_vec, data_quantile, quantile = 0.9):
    distribution = []
    for b in range(5000):
        # Resample points with replacement
        sample_eng = np.random.choice(rt_eng_vec, len(rt_eng_vec))
        # Get sample quantile
        sample_eng_quantile = np.quantile(sample_eng, quantile)
        sample_dis = np.random.choice(rt_dis_vec, len(rt_dis_vec))
        sample_dis_quantile = np.quantile(sample_dis, quantile)
        distribution.append(sample_dis_quantile - sample_eng_quantile)
    # Now return 2.5 and 97.5
    max_val = np.max(distribution)
    min_val = np.min(distribution)
    lower = np.quantile(distribution, 0.025)
    upper = np.quantile(distribution, 0.975)
    frac_above_true = np.sum(distribution >= data_quantile)/len(distribution)
    return lower, upper, min_val, max_val, frac_above_true


def read_bootstrapped_median(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    median, lower, upper, mean_viol_rate_dist = data[0], data[1], data[2], data[3]
    return median, lower, upper, mean_viol_rate_dist


if __name__ == '__main__':
    data_dir = 'data/ibl/data_for_cluster/data_by_animal/'
    animal_list = load_animal_list(data_dir + 'animal_list.npz')
    figure_dir = 'figures/figures_for_paper/figure_6/'
    sigma_val = 2
    covar_set = 2
    alpha_val = 2

    differences_all_animals = []

    animal_list =animal_list[[21, 18, 19, 20, 11, 16 ,24 ,13 ,17 ,10, 22, 31 ,35, 14,  9 ,26 ,23 ,15 ,27 ,33 , 1 , 8 ,25,  3,
  4 , 0 , 6 ,30, 29 ,32 ,28 ,34,  2 , 5, 12, 36 , 7]]]
    fig, ax = plt.subplots(figsize =(1.35,2))
    for z, animal in enumerate(animal_list):
            results_dir =  'results/ibl_individual_fit_old/' + 'covar_set_' + str(covar_set) + '/' + 'prior_sigma_'+ str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/' + animal +'/'

            cv_file = results_dir + "/cvbt_folds_model.npz"
            cvbt_folds_model = load_cv_arr(cv_file)

            K = 3
            with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
                best_init_cvbt_dict = json.load(f)

            # Get the file name corresponding to the best initialization for given K value
            raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K, results_dir, best_init_cvbt_dict)
            hmm_params, lls = load_glmhmm_data(raw_file)

            # Save parameters for initializing individual fits
            weight_vectors = -hmm_params[2]
            log_transition_matrix = hmm_params[1][0]
            init_state_dist = hmm_params[0][0]

            if animal == "ibl_witten_05" or animal == "CSHL_001":
                permutation = calculate_state_permutation(hmm_params)
                weight_vectors = weight_vectors[permutation]
                log_transition_matrix  = permute_transition_matrix(log_transition_matrix , permutation)
            else:
                permutation = range(K)
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

            posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, permutation, alpha_val, sigma_val)
            # Read in RTs
            rts, rts_sess = read_rts('data/ibl/response_times/data_by_animal/'+animal+'.npz')

            rts_engaged = rts[np.where(posterior_probs[:, 0] >= 0.9)[0]]
            rts_engaged = rts_engaged[np.where(~np.isnan(rts_engaged))]
            rts_disengaged = rts[np.where((posterior_probs[:, 1] >= 0.9) | (posterior_probs[:, 2] >= 0.9))[0]]
            rts_disengaged = rts_disengaged[np.where(~np.isnan(rts_disengaged))]

            # Get 90th percentile for each
            quant = 0.90
            eng_quantile = np.quantile(rts_engaged,quant)
            dis_quantile = np.quantile(rts_disengaged, quant)
            diff_quantile = dis_quantile-eng_quantile

            differences_all_animals.append(diff_quantile)
            # Perform bootstrap to get error bars:
            lower_eng, upper_eng, min_val_eng, max_val_eng, frac_above_true = perform_bootstrap(rts_engaged, rts_disengaged, diff_quantile, quant)
            plt.scatter(diff_quantile, z, color = 'r', s=1)
            plt.plot([lower_eng, upper_eng], [z, z], color = 'r', lw = 0.75)
    overall_dir = 'results/ibl_individual_fit/' + 'covar_set_' + str(
        covar_set) +'/'+ '/prior_sigma_' + str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/'
    median, lower, upper, mean_viol_rate_dist = read_bootstrapped_median(overall_dir + 'median_response_bootstrap.npz')
    plt.plot([lower, upper], [z + 1, z + 1], color='#0343df', lw=0.75)
    plt.scatter(median, z + 1, color='b', s=1)
    plt.ylabel("animal (IBL)", fontsize = 10)
    plt.xlabel("$\Delta$ 90th quantile \nresponse time (s) \n (disengaged - \nengaged)", fontsize = 10)
    plt.yticks([])
    plt.xticks([0, 2.5, 5, 7.5, 10, 12.5, 15], ['0', '','5','','10', '', '15'], fontsize = 10)
    plt.axvline(x=0, linestyle = '--', color = 'k', alpha = 0.5, lw=0.75)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.show()
    fig.savefig(figure_dir + 'differences_percentile_response_times_population_bootstrapped_medians_90.pdf')





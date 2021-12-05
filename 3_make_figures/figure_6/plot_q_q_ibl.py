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

import matplotlib as mpl
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

if __name__ == '__main__':
    data_dir = 'data/ibl/data_for_cluster/data_by_animal/'
    figure_dir = '/figures/figures_for_paper/figure_6/'
    animal_list = load_animal_list(data_dir + 'animal_list.npz')
    sigma_val = 2
    covar_set = 2
    alpha_val = 2


    differences_all_animals = []
    se_all_animals = []
    animals_that_reject = []

    fig, ax = plt.subplots(figsize =(2,2))
    response_times_all_animals = []
    for animal in animal_list:
            results_dir ='results/ibl_individual_fit_old/' + 'covar_set_' + str(covar_set) + '/' + 'prior_sigma_'+ str(sigma_val) + '_transition_alpha_' + str(alpha_val) + '/' + animal +'/'

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
                log_transition_matrix = permute_transition_matrix(log_transition_matrix, permutation)
            else:
                permutation = range(K)

            # Also get data for animal:
            inpt, y, session = load_data(data_dir + animal + '_processed.npz')
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

            posterior_probs = get_marginal_posterior(inputs, datas, train_masks, hmm_params, K, permutation, alpha_val, sigma_val)
            # Read in RTs
            rts, rts_sess = read_rts('data/ibl/response_times/data_by_animal/'+animal+'.npz')
            response_times_all_animals.append(rts)

            rts_engaged = rts[np.where(posterior_probs[:, 0] >= 0.9)[0]]
            rts_disengaged = rts[np.where((posterior_probs[:, 1] >= 0.9) | (posterior_probs[:, 2] >= 0.9))[0]]
            max_val = np.nanmax(rts_engaged)
            max_val_2 = np.nanmax(rts_disengaged)
            max_val_overall = np.nanmax([max_val, max_val_2])

            ax.set_xscale("log", nonposx='clip')
            ax.set_yscale("log", nonposy='clip')
            eng_quantiles = []
            dis_quantiles = []
            for i in np.arange(0.01, 1.01, 0.01):
                eng_quantile = np.quantile(rts_engaged[np.where(~np.isnan(rts_engaged))],i)
                dis_quantile = np.quantile(rts_disengaged[np.where(~np.isnan(rts_disengaged))], i)
                eng_quantiles.append(eng_quantile)
                dis_quantiles.append(dis_quantile)
                if i == 0.9 or i == 0.7:
                    col = 'r'
                else:
                    col = 'k'
                if i == 0.9:
                    plt.scatter(eng_quantile, dis_quantile, color = 'r', zorder=2, s = 0.75)
            plt.plot(eng_quantiles, dis_quantiles, 'o-', color = 'grey', alpha = 0.3, linewidth = 0.5, markersize = 0.5, zorder = 0)
            #plt.plot(np.arange(0.01, max_val_overall, max_val_overall/100), np.arange(0.01, max_val_overall, max_val_overall/100), color='k', linestyle='--')
            ax.set_ylim(ymin=0.1)
            ax.set_xlim(xmin=0.1)
            plt.ylabel("disengaged response time (s)", fontsize = 10, labelpad=0)
            plt.xlabel("engaged response time (s)", fontsize = 10, labelpad=0)
            #plt.xlim((0, max_val_overall))
            #plt.ylim((0, max_val_overall))
            from scipy import stats
            _, p = stats.ks_2samp(rts_engaged, rts_disengaged)
            if p >= 0.05:
                print(animal + " does not reject null")
            #plt.title(animal + "; KS-test p = " + str(p))
    plt.plot(np.arange(0.01, max_val_overall, max_val_overall / 100),
             np.arange(0.01, max_val_overall, max_val_overall / 100), color='k', linestyle='--')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.show()
    fig.savefig(figure_dir +'q_q_plot_all_animals_ibl.pdf')


    flattened_list = []
    for x in response_times_all_animals:
        for y in x:
            flattened_list.append(y)
    flattened_list = np.array(flattened_list)

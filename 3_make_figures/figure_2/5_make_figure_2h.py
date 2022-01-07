# Code to produce data for Figure 2h of Ashwood et al. (2020)
# Simulate data from GLM-HMM and calculate psychometric curve for this data
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import seaborn as sns
from scipy.stats import bernoulli
from ssm import HMM

sys.path.insert(0, '../')
from plotting_utils import load_glmhmm_data, load_data, load_cv_arr, \
    load_reward_data, get_file_name_for_best_model_fold, \
    partition_data_by_session, create_violation_mask, calculate_correct_ans,\
    get_prob_right

D = 1  # data dimension
C = 2  # number of output types/categories

npr.seed(0)


if __name__ == '__main__':
    animal = "CSHL_008"
    K = 3
    sigma_val = 2
    covar_set = 2
    alpha_val = 2

    data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    results_dir = '../../results/ibl_individual_fit/' + animal + '/'
    figure_dir = '../../figures/figure_2/'

    cv_file = results_dir + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    # Also get data for animal:
    inpt, y, session = load_data(data_dir + animal + '_processed.npz')
    rewarded = load_reward_data(data_dir + animal + '_rewarded.npz')
    correct_answer = calculate_correct_ans(y, rewarded)
    T = inpt.shape[0]

    violation_idx = np.where(y == -1)[0]
    non_viol_mask = (y != -1) + 0
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])
    y[np.where(y == -1), :] = 1

    inputs, datas, train_masks = partition_data_by_session(
        np.hstack((inpt, np.ones((len(inpt), 1)))), y, mask, session)

    # Get HMM params:
    with open(results_dir + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K
    # value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                 results_dir,
                                                 best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    weight_vectors = hmm_params[2]

    D, M, C = 1, 4, 2
    this_hmm = HMM(K,
                   D,
                   M,
                   observations="input_driven_obs",
                   observation_kwargs=dict(C=C),
                   transitions="standard")
    this_hmm.params = hmm_params
    # Now sample a y and z from this GLM-HMM:
    latents = []
    datas = []
    trial_num = 0

    stim_vals, _ = get_prob_right(-weight_vectors, inpt, 0, 1, 1)

    psychometric_grid = []

    for i, input in enumerate(inputs):
        T = input.shape[0]
        # Sample a value for t = 0 value of past choice covariate
        pc = (2 * bernoulli.rvs(0.5, size=1)) - 1
        input[0, 1] = pc
        # Sample a value for t = 0 value of wsls covariate
        wsls = (2 * bernoulli.rvs(0.5, size=1)) - 1
        input[0, 2] = wsls
        latent_z = np.zeros(input.shape[0], dtype=int)
        data = np.zeros(input.shape[0], dtype=int)

        # Now loop through each time and get the state and the observation
        # for each time step:
        pi0 = np.exp(hmm_params[0][0])
        latent_z[0] = int(npr.choice(K, p=pi0))

        # Get psychometric for each state:
        psychometric_this_t = np.zeros((K, len(stim_vals)))
        for k in range(K):
            _, psychometric_k_t = get_prob_right(-weight_vectors, inpt, k, pc,
                                                 wsls)
            psychometric_this_t[k] = pi0[k] * np.array(psychometric_k_t)
        psychometric_grid.append(np.sum(psychometric_this_t, axis=0))

        for t in range(0, T):
            Pt = np.exp(hmm_params[1][0])

            this_input = np.expand_dims(input[t], axis=0)
            # Get observation at current trial (based on state)
            data[t] = this_hmm.observations.sample_x(z=latent_z[t],
                                                     xhist=None,
                                                     input=np.expand_dims(
                                                         input[t], axis=0))

            # Get state at next trial
            if t < T - 1:
                latent_z[t + 1] = int(npr.choice(K, p=Pt[latent_z[t]]))
                # update past choice and wsls based on sampled y and correct
                # answer
                input[t + 1, 1] = 2 * data[t] - 1
                inpt[trial_num + 1, 1] = 2 * data[t] - 1
                # wsls:
                rewarded = 2 * (data[t] == correct_answer[trial_num]) - 1
                input[t + 1, 2] = input[t + 1, 1] * rewarded
                inpt[trial_num + 1, 2] = input[t + 1, 1] * rewarded
                # Get psychometric for each state:
                psychometric_this_t = np.zeros((K, len(stim_vals)))
                for k in range(K):
                    _, psychometric_k_t = get_prob_right(
                        -weight_vectors, inpt, k, 2 * data[t] - 1,
                        (2 * data[t] - 1) * rewarded)
                    psychometric_this_t[k] = Pt[latent_z[t]][k] * np.array(
                        psychometric_k_t)
                psychometric_grid.append(np.sum(psychometric_this_t, axis=0))

            trial_num += 1

        latents.append(latent_z)
        datas.append(data)
    latents_flattened = np.concatenate(latents)
    datas_flattened = np.concatenate(datas)
    assert trial_num == inpt.shape[0]

    # ============== PLOTTING CODE ===============
    fig = plt.figure(figsize=(1.6, 1.6))
    plt.subplots_adjust(left=0.35, bottom=0.3, right=0.92, top=0.95)
    inpt_df = pd.DataFrame({
        'signed_contrast': inpt[:, 0],
        'choice': y[nonviolation_idx, 0]
    })
    plt.plot(stim_vals,
             np.mean(psychometric_grid, axis=0),
             '-',
             color='k',
             linewidth=0.9)
    sns.lineplot(inpt_df['signed_contrast'],
                 inpt_df['choice'],
                 err_style="bars",
                 linewidth=0,
                 linestyle='None',
                 mew=0,
                 marker='o',
                 markersize=2,
                 ci=95,
                 err_kws={"linewidth": 0.75},
                 zorder=3,
                 color=(193 / 255, 39 / 255, 45 / 255))
    plt.xticks([min(stim_vals), 0, max(stim_vals)],
               labels=['-100', '0', '100'],
               fontsize=10)
    plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
    plt.ylabel('p("R")', fontsize=10)
    plt.xlabel('stimulus', fontsize=10)
    plt.title("all trials", fontsize=10)
    plt.axhline(y=0.5, color="k", alpha=0.45, ls=":", linewidth=0.5)
    plt.axvline(x=0, color="k", alpha=0.45, ls=":", linewidth=0.5)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.ylim((-0.01, 1.01))
    fig.savefig(figure_dir + 'fig2h.pdf')

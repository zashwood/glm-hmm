# Fit lapse model to IBL data
import os

import autograd.numpy as np
import matplotlib.pyplot as plt
from LapseModel import lapse_model
from lapse_utils import load_session_fold_lookup, load_data, \
    get_parmin, get_parmax, get_parstart, calculate_std

np.random.seed(65)

if __name__ == '__main__':

    data_dir = '../../data/ibl/data_for_cluster/'
    results_dir = '../../results/ibl_global_fit/'

    num_lapse_params = 1
    num_folds = 5

    # Fit GLM to all data
    animal_file = data_dir + 'all_animals_concat.npz'
    session_fold_lookup_table = load_session_fold_lookup(
        data_dir + 'all_animals_concat_session_fold_lookup.npz')

    for fold in range(num_folds):
        inpt, y, session = load_data(animal_file)
        labels_for_plot = ['flashes', 'P_C', 'WSLS', 'bias']
        y = y.astype('int')

        sessions_to_keep = session_fold_lookup_table[np.where(
            session_fold_lookup_table[:, 1] != fold), 0]
        idx_this_fold = [
            str(sess) in sessions_to_keep and y[id, 0] != -1
            for id, sess in enumerate(session)
        ]
        this_inpt, this_y, this_session = inpt[idx_this_fold, :], \
                                          y[idx_this_fold, :], \
                                          session[idx_this_fold]
        train_size = this_inpt.shape[0]
        M = this_inpt.shape[1]
        loglikelihood_train_vector = []

        figure_directory = results_dir + "Lapse_Model/fold_" + str(fold) + '/'
        if not os.path.exists(figure_directory):
            os.makedirs(figure_directory)

        M = this_inpt.shape[1]
        loglikelihood_train_vector = []

        # Create parameter grid to search over when using multiple
        # initializations
        parmin_grid = np.array(
            [get_parmin(i, M) for i in range(M + 1 + num_lapse_params)])
        parmax_grid = np.array(
            [get_parmax(i, M) for i in range(M + 1 + num_lapse_params)])
        parstart_grid = np.array(
            [get_parstart(i, M) for i in range(M + 1 + num_lapse_params)])

        # structures to use for saving data
        num_initializations = 10
        log_likelihoods = np.zeros(num_initializations, )
        pars = []
        hessians = []

        for init in range(num_initializations):
            print("Current initialization = " + str(init))
            parstart = parmin_grid + np.random.rand(
                parmin_grid.size) * (parmax_grid - parmin_grid)
            # Instantiate new model
            new_model = lapse_model(M, num_lapse_params)
            # Initialize parameters as parstart
            new_model.params = [parstart[range(M + 1)], parstart[(M + 1):]]
            # Fit model, and obtain loglikelihood and parameters
            new_model.fit_lapse_model(datas=[this_y],
                                      inputs=[this_inpt],
                                      masks=None,
                                      tags=None)
            # Loglikelihood
            final_ll = new_model.log_marginal(datas=[this_y],
                                              inputs=[this_inpt],
                                              masks=None,
                                              tags=None)
            log_likelihoods[init] = final_ll
            # Parameters
            pars.append([new_model.params])
            hessians.append([new_model.hessian])

        ordered_initializations = np.argsort(-log_likelihoods)

        # Plot recovered GLM weights and lapse parameter for top 3
        # initializations:
        fig = plt.figure(figsize=(4 * (4 + num_lapse_params), 8),
                         dpi=80,
                         facecolor='w',
                         edgecolor='k')
        plt.subplots_adjust(left=0.05,
                            bottom=0.12,
                            right=0.95,
                            top=0.85,
                            wspace=0.3,
                            hspace=0.3)

        lapse_params_for_plotting = []
        lapse_std = []
        glm_weights_for_plotting = []

        for j, init in enumerate(ordered_initializations[range(3)]):
            print("best init: " + str(init))
            plt.subplot(1, 4 + num_lapse_params, j + 1)
            these_pars = pars[init]
            glm_weights = these_pars[0][0]
            # Get hessian
            this_hess = hessians[init][0]
            # Get sd
            this_sd = calculate_std(this_hess)
            # Save lapse parameters for plotting next:
            this_ll = log_likelihoods[init]
            if num_lapse_params == 1:
                this_lapse = these_pars[0][1][0]
                lapse_std.append(this_sd[-1])
                if j == 0:
                    np.savez(figure_directory +
                             'lapse_model_params_one_param.npz',
                             loglikelihood=this_ll,
                             glm_weights=glm_weights,
                             glm_std=this_sd,
                             lapse=this_lapse,
                             lapse_std=lapse_std)
                    glm_weights_for_plotting.append(glm_weights)
            else:
                this_lapse = [these_pars[0][1][0], these_pars[0][1][1]]
                lapse_std.append([this_sd[-2], this_sd[-1]])
                if j == 0:
                    np.savez(figure_directory +
                             'lapse_model_params_two_param.npz',
                             loglikelihood=this_ll,
                             glm_weights=glm_weights,
                             glm_std=this_sd,
                             lapse=this_lapse,
                             lapse_std=lapse_std)
                    glm_weights_for_plotting.append(glm_weights)
            lapse_params_for_plotting.append(this_lapse)

            # Plot GLM weights
            plt.plot(glm_weights + this_sd[range(len(glm_weights))],
                     marker='o',
                     color='b')
            plt.plot(glm_weights - this_sd[range(len(glm_weights))],
                     marker='o',
                     color='b')
            plt.fill_between(range(len(glm_weights)),
                             glm_weights + this_sd[range(len(glm_weights))],
                             glm_weights - this_sd[range(len(glm_weights))],
                             alpha=0.35,
                             color='blue')
            plt.plot(glm_weights,
                     label="Init " + str(init),
                     alpha=0.5,
                     color='b')
            plt.axhline(y=0, color="k", alpha=0.5, ls="--")
            plt.xticks(list(range(0, len(labels_for_plot))),
                       labels_for_plot,
                       rotation='45',
                       fontsize=12)
            plt.title("GLM Weights; Initialization " + str(init) +
                      "\n Loglikelihood = " + str(this_ll))

        if num_lapse_params == 1:
            lapse_params_for_plotting = np.array(lapse_params_for_plotting)
            # lapse_std = np.array(lapse_std)
            plt.subplot(1, 3 + num_lapse_params, 4)
            plt.plot(lapse_params_for_plotting, color='r')
            plt.plot(lapse_params_for_plotting + lapse_std, color='r')
            plt.plot(lapse_params_for_plotting - lapse_std, color='r')
            plt.fill_between(range(len(lapse_params_for_plotting)),
                             lapse_params_for_plotting + lapse_std,
                             lapse_params_for_plotting - lapse_std,
                             alpha=0.35,
                             color='red')
            plt.title("Lapse Parameters")
            plt.xlabel("Ranked Initialization")
            plt.ylabel("Lapse Parameter")
            fig.suptitle(
                "Logistic Regression with Lapse Parameter: Top 3 "
                "Initializations \n All Animals",
                fontsize=15)
            fig.savefig(figure_directory + 'lapse_model_fit_1_param.png')

        else:
            lapse_params_for_plotting = np.array(lapse_params_for_plotting)
            lapse_std = np.array(lapse_std)
            # lapse_std = np.array(lapse_std)
            plt.subplot(1, 3 + num_lapse_params, 4)
            plt.plot(lapse_params_for_plotting[:, 0], color='r')
            plt.plot(lapse_params_for_plotting[:, 0] + lapse_std[:, 0],
                     color='r')
            plt.plot(lapse_params_for_plotting[:, 0] - lapse_std[:, 0],
                     color='r')
            plt.fill_between(range(lapse_params_for_plotting.shape[0]),
                             lapse_params_for_plotting[:, 0] + lapse_std[:, 0],
                             lapse_params_for_plotting[:, 0] - lapse_std[:, 0],
                             alpha=0.35,
                             color='red')
            plt.title("Lapse Parameters")
            plt.xlabel("Ranked Initialization")
            plt.ylabel("Lapse Parameter 1")

            plt.subplot(1, 3 + num_lapse_params, 5)
            plt.plot(lapse_params_for_plotting[:, 1], color='r')
            plt.plot(lapse_params_for_plotting[:, 1] + lapse_std[:, 1],
                     color='r')
            plt.plot(lapse_params_for_plotting[:, 1] - lapse_std[:, 1],
                     color='r')
            plt.fill_between(range(lapse_params_for_plotting.shape[0]),
                             lapse_params_for_plotting[:, 1] + lapse_std[:, 1],
                             lapse_params_for_plotting[:, 1] - lapse_std[:, 1],
                             alpha=0.35,
                             color='red')
            plt.title("Lapse Parameters")
            plt.xlabel("Ranked Initialization")
            plt.ylabel("Lapse Parameter 2")

        if num_lapse_params == 1:
            fig.suptitle(
                "Logistic Regression with Lapse Parameter: Top 3 "
                "Initializations \n All Animals",
                fontsize=15)
            fig.savefig(figure_directory + 'lapse_model_fit_1_param.png')
        else:
            fig.suptitle(
                "Logistic Regression with Lapse Parameter: Top 3 "
                "Initializations \n All Animals",
                fontsize=15)
            fig.savefig(figure_directory + 'lapse_model_fit_2_param.png')

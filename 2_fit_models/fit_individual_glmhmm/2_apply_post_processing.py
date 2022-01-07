# Create a matrix of size num_models x num_folds containing normalized
# loglikelihood values for train and test sets

import json
import sys
import numpy as np

sys.path.insert(0, '../fit_global_glmhmm/')
from glm_hmm_utils import load_animal_list
from post_processing_utils import load_data, load_session_fold_lookup, \
    prepare_data_for_cv, calculate_baseline_test_ll, \
    calculate_glm_test_loglikelihood, calculate_cv_bit_trial, \
    return_glmhmm_nll, return_lapse_nll

if __name__ == '__main__':

    prior_sigma = 2
    transition_alpha = 2

    data_dir = '../../data/ibl/data_for_cluster/data_by_animal/'
    results_dir = '../../results/ibl_individual_fit/'

    # Parameters
    C = 2  # number of output classes
    num_folds = 5  # number of folds
    D = 1  # number of output dimensions
    K_max = 5  # number of latent states
    num_models = K_max + 2  # model for each latent + 2 lapse
    # models

    animal_list = load_animal_list(data_dir + 'animal_list.npz')
    for animal in animal_list:
        overall_dir = results_dir + animal + '/'
        # Load data
        inpt, y, session = load_data(data_dir + animal + '_processed.npz')
        session_fold_lookup_table = load_session_fold_lookup(
            data_dir + animal + '_session_fold_lookup.npz')

        animal_preferred_model_dict = {}
        models = ["GLM", "Lapse_Model", "GLM_HMM"]

        cvbt_folds_model = np.zeros((num_models, num_folds))
        cvbt_train_folds_model = np.zeros((num_models, num_folds))

        # Save best initialization for each model-fold combination
        best_init_cvbt_dict = {}
        for fold in range(num_folds):
            print("fold = " + str(fold))
            test_inpt, test_y, test_nonviolation_mask, \
            this_test_session, train_inpt, train_y, \
            train_nonviolation_mask, this_train_session, M, n_test, \
            n_train = prepare_data_for_cv(
                inpt, y, session, session_fold_lookup_table, fold)
            ll0 = calculate_baseline_test_ll(
                train_y[train_nonviolation_mask == 1, :],
                test_y[test_nonviolation_mask == 1, :], C)
            ll0_train = calculate_baseline_test_ll(
                train_y[train_nonviolation_mask == 1, :],
                train_y[train_nonviolation_mask == 1, :], C)
            for model in models:
                print("model = " + str(model))
                if model == "GLM":
                    # Load parameters and instantiate a new GLM
                    # object with these parameters
                    glm_weights_file = overall_dir + \
                                       '/GLM/fold_' + str(
                        fold) + '/variables_of_interest_iter_0.npz'
                    ll_glm = calculate_glm_test_loglikelihood(
                        glm_weights_file,
                        test_y[test_nonviolation_mask == 1, :],
                        test_inpt[test_nonviolation_mask == 1, :], M, C)
                    ll_glm_train = calculate_glm_test_loglikelihood(
                        glm_weights_file,
                        train_y[train_nonviolation_mask == 1, :],
                        train_inpt[train_nonviolation_mask == 1, :], M, C)
                    cvbt_folds_model[0, fold] = calculate_cv_bit_trial(
                        ll_glm, ll0, n_test)
                    cvbt_train_folds_model[0, fold] = calculate_cv_bit_trial(
                        ll_glm_train, ll0_train, n_train)
                elif model == "Lapse_Model":
                    # One lapse parameter model:
                    cvbt_folds_model[1, fold], cvbt_train_folds_model[
                        1, fold], _, _ = return_lapse_nll(
                            inpt, y, session, session_fold_lookup_table, fold,
                            1, overall_dir, C)
                    # Two lapse parameter model:
                    cvbt_folds_model[2, fold], cvbt_train_folds_model[
                        2, fold], _, _ = return_lapse_nll(
                            inpt, y, session, session_fold_lookup_table, fold,
                            2, overall_dir, C)
                elif model == "GLM_HMM":
                    for K in range(2, K_max+1):
                        print("K = "+ str(K))
                        model_idx = 3 + (K - 2)
                        cvbt_folds_model[model_idx, fold], \
                        cvbt_train_folds_model[
                            model_idx, fold], _, _, \
                        init_ordering_by_train = return_glmhmm_nll(
                            np.hstack((inpt, np.ones((len(inpt), 1)))), y, session,
                            session_fold_lookup_table, fold, K, D, C,
                            overall_dir)
                        # Save best initialization to dictionary for
                        # later:
                        key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(
                            fold)
                        best_init_cvbt_dict[key_for_dict] = int(
                            init_ordering_by_train[0])
        # Save best initialization directories across animals,
        # folds and models (only GLM-HMM):
        print(cvbt_folds_model)
        print(cvbt_train_folds_model)
        json_dump = json.dumps(best_init_cvbt_dict)
        f = open(overall_dir + "/best_init_cvbt_dict.json", "w")
        f.write(json_dump)
        f.close()
        # Save cvbt_folds_model as numpy array for easy parsing
        # across all models and folds
        np.savez(overall_dir + "/cvbt_folds_model.npz", cvbt_folds_model)
        np.savez(overall_dir + "/cvbt_train_folds_model.npz",
                 cvbt_train_folds_model)
#  Functions to assist with post-processing of GLM-HMM fits
import glob
import re
import sys

import numpy as np
import pandas as pd
import ssm

sys.path.insert(0, '../fit_glm/')
sys.path.insert(0, '../fit_lapse_model/')
from GLM import glm
from LapseModel import lapse_model


def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    y = y.astype('int')
    session = data[2]
    return inpt, y, session


def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table


def load_glm_vectors(glm_vectors_file):
    container = np.load(glm_vectors_file)
    data = [container[key] for key in container]
    loglikelihood_train = data[0]
    recovered_weights = data[1]
    return loglikelihood_train, recovered_weights


def load_lapse_params(lapse_file):
    container = np.load(lapse_file, allow_pickle=True)
    data = [container[key] for key in container]
    lapse_loglikelihood = data[0]
    lapse_glm_weights = data[1]
    lapse_glm_weights_std = data[2],
    lapse_p = data[3]
    lapse_p_std = data[4]
    return lapse_loglikelihood, lapse_glm_weights, lapse_glm_weights_std, \
           lapse_p, lapse_p_std


def load_glmhmm_data(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    this_hmm_params = data[0]
    lls = data[1]
    return [this_hmm_params, lls]


def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model


def partition_data_by_session(inpt, y, mask, session):
    '''
    Partition inpt, y, mask by session
    :param inpt: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or
    not
    :param session: list of size T containing session ids
    :return: list of inpt arrays, data arrays and mask arrays, where the
    number of elements in list = number of sessions and each array size is
    number of trials in session
    '''
    inputs = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [
        session[index] for index in sorted(indexes)
    ]  # ensure that unique sessions are ordered as they are in
    # session (so we can map inputs back to inpt)
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(inpt[idx, :])
        datas.append(y[idx, :])
        masks.append(mask[idx])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks


def get_train_test_dta(inpt, y, mask, session, session_fold_lookup_table,
                       fold):
    '''
    Split inpt, y, mask, session arrays into train and test arrays
    :param inpt:
    :param y:
    :param mask:
    :param session:
    :param session_fold_lookup_table:
    :param fold:
    :return:
    '''
    test_sessions = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] == fold), 0]
    train_sessions = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] != fold), 0]
    idx_test = [str(sess) in test_sessions for sess in session]
    idx_train = [str(sess) in train_sessions for sess in session]
    test_inpt, test_y, test_mask, this_test_session = inpt[idx_test, :], y[
                                                                         idx_test,
                                                                         :], \
                                                      mask[idx_test], session[
                                                          idx_test]
    train_inpt, train_y, train_mask, this_train_session = inpt[idx_train,
                                                          :], y[idx_train,
                                                              :], \
                                                          mask[idx_train], \
                                                          session[idx_train]
    return test_inpt, test_y, test_mask, this_test_session, train_inpt, \
           train_y, train_mask, this_train_session


def create_violation_mask(violation_idx, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1
    = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in violation_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonviolation_idx) + len(
        violation_idx) == T, "violation and non-violation idx do not include " \
                             "" \
                             "" \
                             "" \
                             "" \
                             "all dta!"
    return nonviolation_idx, mask


def prepare_data_for_cv(inpt, y, session, session_fold_lookup_table, fold):
    '''
    :return:
    '''

    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, nonviolation_mask = create_violation_mask(
        violation_idx, inpt.shape[0])
    # Load train and test data for session
    test_inpt, test_y, test_nonviolation_mask, this_test_session, \
    train_inpt, train_y, train_nonviolation_mask, this_train_session = \
        get_train_test_dta(
            inpt, y, nonviolation_mask, session, session_fold_lookup_table,
            fold)
    M = train_inpt.shape[1]
    n_test = np.sum(test_nonviolation_mask == 1)
    n_train = np.sum(train_nonviolation_mask == 1)
    return test_inpt, test_y, test_nonviolation_mask, this_test_session, \
           train_inpt, train_y, train_nonviolation_mask, this_train_session, \
           M, n_test, n_train


def calculate_baseline_test_ll(train_y, test_y, C):
    """
    Calculate baseline loglikelihood for CV bit/trial calculation.  This is
    log(p(y|p0)) = n_right(log(p0)) + (n_total-n_right)log(1-p0), where p0
    is the proportion of trials
    in which the animal went right in the training set and n_right is the
    number of trials in which the animal went right in the test set
    :param train_y
    :param test_y
    :return: baseline loglikelihood for CV bit/trial calculation
    """
    _, train_class_totals = np.unique(train_y, return_counts=True)
    train_class_probs = train_class_totals / train_y.shape[0]
    _, test_class_totals = np.unique(test_y, return_counts=True)
    ll0 = 0
    for c in range(C):
        ll0 += test_class_totals[c] * np.log(train_class_probs[c])
    return ll0


def calculate_glm_test_loglikelihood(glm_weights_file, test_y, test_inpt, M,
                                     C):
    loglikelihood_train, glm_vectors = load_glm_vectors(glm_weights_file)
    # Calculate test loglikelihood
    new_glm = glm(M, C)
    # Set parameters to fit parameters:
    new_glm.params = glm_vectors
    # Get loglikelihood of training data:
    loglikelihood_test = new_glm.log_marginal([test_y], [test_inpt], None,
                                              None)
    return loglikelihood_test


def calculate_lapse_test_loglikelihood(lapse_file, test_y, test_inpt, M,
                                       num_lapse_params):
    lapse_loglikelihood, lapse_glm_weights, _, lapse_p, _ = load_lapse_params(
        lapse_file)
    # Instantiate a model with these parameters
    new_lapse_model = lapse_model(M, num_lapse_params)
    if num_lapse_params == 1:
        new_lapse_model.params = [lapse_glm_weights, np.array([lapse_p])]
    else:
        new_lapse_model.params = [lapse_glm_weights, lapse_p]
    # Now calculate test loglikelihood
    loglikelihood_test = new_lapse_model.log_marginal(datas=[test_y],
                                                      inputs=[test_inpt],
                                                      masks=None,
                                                      tags=None)
    return loglikelihood_test


def return_lapse_nll(inpt, y, session, session_fold_lookup_table, fold,
                     num_lapse_params, results_dir_glm_lapse, C):
    test_inpt, test_y, test_nonviolation_mask, this_test_session, \
    train_inpt, train_y, train_nonviolation_mask, this_train_session, M, \
    n_test, n_train = prepare_data_for_cv(
        inpt, y, session, session_fold_lookup_table, fold)
    ll0 = calculate_baseline_test_ll(train_y[train_nonviolation_mask == 1, :],
                                     test_y[test_nonviolation_mask == 1, :], C)
    ll0_train = calculate_baseline_test_ll(
        train_y[train_nonviolation_mask == 1, :],
        train_y[train_nonviolation_mask == 1, :], C)
    if num_lapse_params == 1:
        lapse_file = results_dir_glm_lapse + '/Lapse_Model/fold_' + str(
            fold) + '/lapse_model_params_one_param.npz'
    elif num_lapse_params == 2:
        lapse_file = results_dir_glm_lapse + '/Lapse_Model/fold_' + str(
            fold) + '/lapse_model_params_two_param.npz'
    ll_lapse = calculate_lapse_test_loglikelihood(
        lapse_file,
        test_y[test_nonviolation_mask == 1, :],
        test_inpt[test_nonviolation_mask == 1, :],
        M,
        num_lapse_params=num_lapse_params)
    ll_train_lapse = calculate_lapse_test_loglikelihood(
        lapse_file,
        train_y[train_nonviolation_mask == 1, :],
        train_inpt[train_nonviolation_mask == 1, :],
        M,
        num_lapse_params=num_lapse_params)
    nll_lapse = calculate_cv_bit_trial(ll_lapse, ll0, n_test)
    nll_lapse_train = calculate_cv_bit_trial(ll_train_lapse, ll0_train,
                                             n_train)
    return nll_lapse, nll_lapse_train, ll_lapse, ll_train_lapse


def calculate_glm_hmm_test_loglikelihood(glm_hmm_dir, test_datas, test_inputs,
                                         test_nonviolation_masks, K, D, M, C):
    """
    calculate test loglikelihood for GLM-HMM model.  Loop through all
    initializations for fold of interest, and check that final train LL is
    same for top initializations
    :return:
    """
    this_file_name = glm_hmm_dir + '/iter_*/glm_hmm_raw_parameters_*.npz'
    raw_files = glob.glob(this_file_name, recursive=True)
    train_ll_vals_across_iters = []
    test_ll_vals_across_iters = []
    for file in raw_files:
        # Loop through initializations and calculate BIC:
        this_hmm_params, lls = load_glmhmm_data(file)
        train_ll_vals_across_iters.append(lls[-1])
        # Instantiate a new HMM and calculate test loglikelihood:
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C),
                           transitions="standard")
        this_hmm.params = this_hmm_params
        test_ll = this_hmm.log_likelihood(test_datas,
                                          inputs=test_inputs,
                                          masks=test_nonviolation_masks)
        test_ll_vals_across_iters.append(test_ll)
    # Order initializations by train LL (don't train on test data!):
    train_ll_vals_across_iters = np.array(train_ll_vals_across_iters)
    test_ll_vals_across_iters = np.array(test_ll_vals_across_iters)
    # Order raw files by train LL
    file_ordering_by_train = np.argsort(-train_ll_vals_across_iters)
    raw_file_ordering_by_train = np.array(raw_files)[file_ordering_by_train]
    # Get initialization number from raw_file ordering
    init_ordering_by_train = [
        int(re.findall(r'\d+', file)[-1])
        for file in raw_file_ordering_by_train
    ]
    return test_ll_vals_across_iters, init_ordering_by_train, \
           file_ordering_by_train


def return_glmhmm_nll(inpt, y, session, session_fold_lookup_table, fold, K, D,
                      C, results_dir_glm_hmm):
    '''
    For a given fold, return NLL for both train and test datasets for
    GLM-HMM model with K, D, C.  Requires reading in best
    parameters over all initializations for GLM-HMM (hence why
    results_dir_glm_hmm is required as an input)
    :param inpt:
    :param y:
    :param session:
    :param session_fold_lookup_table:
    :param fold:
    :param K:
    :param D:
    :param C:
    :param results_dir_glm_hmm:
    :return:
    '''
    test_inpt, test_y, test_nonviolation_mask, this_test_session, \
    train_inpt, train_y, train_nonviolation_mask, this_train_session, M, \
    n_test, n_train = prepare_data_for_cv(
        inpt, y, session, session_fold_lookup_table, fold)
    ll0 = calculate_baseline_test_ll(train_y[train_nonviolation_mask == 1, :],
                                     test_y[test_nonviolation_mask == 1, :], C)
    ll0_train = calculate_baseline_test_ll(
        train_y[train_nonviolation_mask == 1, :],
        train_y[train_nonviolation_mask == 1, :], C)
    # For GLM-HMM set values of y for violations to 1.  This value doesn't
    # matter (as mask will ensure that these y values do not contribute to
    # loglikelihood calculation
    test_y[test_nonviolation_mask == 0, :] = 1
    train_y[train_nonviolation_mask == 0, :] = 1
    # For GLM-HMM, need to partition data by session
    test_inputs, test_datas, test_nonviolation_masks = \
        partition_data_by_session(
            test_inpt, test_y,
            np.expand_dims(test_nonviolation_mask, axis=1),
            this_test_session)
    train_inputs, train_datas, train_nonviolation_masks = \
        partition_data_by_session(
            train_inpt, train_y,
            np.expand_dims(train_nonviolation_mask, axis=1),
            this_train_session)
    dir_to_check = results_dir_glm_hmm + '/GLM_HMM_K_' + str(
        K) + '/fold_' + str(fold) + '/'
    test_ll_vals_across_iters, init_ordering_by_train, \
    file_ordering_by_train = calculate_glm_hmm_test_loglikelihood(
        dir_to_check, test_datas, test_inputs, test_nonviolation_masks, K, D,
        M, C)
    train_ll_vals_across_iters, _, _ = calculate_glm_hmm_test_loglikelihood(
        dir_to_check, train_datas, train_inputs, train_nonviolation_masks, K,
        D, M, C)
    test_ll_vals_across_iters = test_ll_vals_across_iters[
        file_ordering_by_train]
    train_ll_vals_across_iters = train_ll_vals_across_iters[
        file_ordering_by_train]
    ll_glm_hmm_this_K = test_ll_vals_across_iters[0]
    cvbt_thismodel_thisfold = calculate_cv_bit_trial(ll_glm_hmm_this_K, ll0,
                                                     n_test)
    train_cvbt_thismodel_thisfold = calculate_cv_bit_trial(
        train_ll_vals_across_iters[0], ll0_train, n_train)
    return cvbt_thismodel_thisfold, train_cvbt_thismodel_thisfold, \
           ll_glm_hmm_this_K, \
           train_ll_vals_across_iters[0], init_ordering_by_train


def calculate_cv_bit_trial(ll_model, ll_0, n_trials):
    cv_bit_trial = ((ll_model - ll_0) / n_trials) / np.log(2)
    return cv_bit_trial


def create_cv_frame_for_plotting(cv_file):
    cvbt_folds_model = load_cv_arr(cv_file)
    glm_lapse_model = cvbt_folds_model[:3, ]
    idx = np.array([0, 3, 4, 5, 6])
    cvbt_folds_model = cvbt_folds_model[idx, :]
    # Identify best cvbt:
    mean_cvbt = np.mean(cvbt_folds_model, axis=1)
    loc_best = np.where(mean_cvbt == max(mean_cvbt))[0]
    best_val = max(mean_cvbt)
    # Create dataframe for plotting
    num_models = cvbt_folds_model.shape[0]
    num_folds = cvbt_folds_model.shape[1]
    # Create pandas dataframe:
    data_for_plotting_df = pd.DataFrame({
        'model':
            np.repeat(np.arange(num_models), num_folds),
        'cv_bit_trial':
            cvbt_folds_model.flatten()
    })
    return data_for_plotting_df, loc_best, best_val, glm_lapse_model


def get_file_name_for_best_model_fold(cvbt_folds_model, K, overall_dir,
                                      best_init_cvbt_dict):
    '''
    Get the file name for the best initialization for the K value specified
    :param cvbt_folds_model:
    :param K:
    :param models:
    :param overall_dir:
    :param best_init_cvbt_dict:
    :return:
    '''
    # Identify best fold for best model:
    # loc_best = K - 1
    loc_best = 0
    best_fold = np.where(cvbt_folds_model[loc_best, :] == max(cvbt_folds_model[
                                                              loc_best, :]))[
        0][0]
    base_path = overall_dir + '/GLM_HMM_K_' + str(K) + '/fold_' + str(
        best_fold)
    key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    best_iter = best_init_cvbt_dict[key_for_dict]
    raw_file = base_path + '/iter_' + str(
        best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return raw_file


def permute_transition_matrix(transition_matrix, permutation):
    transition_matrix = transition_matrix[np.ix_(permutation, permutation)]
    return transition_matrix


def calculate_state_permutation(hmm_params):
    '''
    If K = 3, calculate the permutation that results in states being ordered
    as engaged/bias left/bias right
    Else: order states so that they are ordered by engagement
    :param hmm_params:
    :return: permutation
    '''
    # GLM weights (note: we have to take negative, because we are interested
    # in weights corresponding to p(y = 1) = 1/(1+e^(-w.x)), but returned
    # weights from
    # code are w such that p(y = 1) = e(w.x)/1+e(w.x))
    glm_weights = -hmm_params[2]
    K = glm_weights.shape[0]
    if K == 3:
        # want states ordered as engaged/bias left/bias right
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = \
            np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged to have a bias
        # that will not cause it to have largest bias
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_left_loc = \
            np.where(
                (reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[
                0][0]
        state_order = [engaged_loc, bias_left_loc]
        bias_right_loc = np.arange(3)[np.where(
            [range(3)[i] not in state_order for i in range(3)])][0]
        permutation = np.array([engaged_loc, bias_left_loc, bias_right_loc])
    elif K == 4:
        # want states ordered as engaged/bias left/bias right
        M = glm_weights.shape[2] - 1
        # bias coefficient is last entry in dimension 2
        engaged_loc = \
            np.where((glm_weights[:, 0, 0] == max(glm_weights[:, 0, 0])))[0][0]
        reduced_weights = np.copy(glm_weights)
        # set row in reduced weights corresponding to engaged to have a bias
        # that will not
        reduced_weights[engaged_loc, 0, M] = max(glm_weights[:, 0, M]) - 0.001
        bias_right_loc = \
            np.where(
                (reduced_weights[:, 0, M] == max(reduced_weights[:, 0, M])))[
                0][0]
        bias_left_loc = \
            np.where(
                (reduced_weights[:, 0, M] == min(reduced_weights[:, 0, M])))[
                0][0]
        state_order = [engaged_loc, bias_left_loc, bias_right_loc]
        other_loc = np.arange(4)[np.where(
            [range(4)[i] not in state_order for i in range(4)])][0]
        permutation = np.array(
            [engaged_loc, bias_left_loc, bias_right_loc, other_loc])
    else:
        # order states by engagement: with the most engaged being first.
        # Note: argsort sorts inputs from smallest to largest (hence why we
        # convert to -ve glm_weights)
        permutation = np.argsort(-glm_weights[:, 0, 0])
    # assert that all indices are present in permutation exactly once:
    assert len(permutation) == K, "permutation is incorrect size"
    assert check_all_indices_present(permutation, K), "not all indices " \
                                                      "present in " \
                                                      "permutation: " \
                                                      "permutation = " + \
                                                      str(permutation)
    return permutation


def check_all_indices_present(permutation, K):
    for i in range(K):
        if i not in permutation:
            return False
    return True


def get_marginal_posterior(inputs, datas, masks, hmm_params, K, permutation):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    this_hmm = ssm.HMM(K, D, M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2),
                       transitions="standard")
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [this_hmm.expected_states(data=data, input=input,
                                             mask=np.expand_dims(mask,
                                                                 axis=1))[0]
                    for data, input, mask
                    in zip(datas, inputs, masks)]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, permutation]
    return posterior_probs

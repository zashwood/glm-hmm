import json

import numpy as np
import numpy.random as npr
import ssm
from scipy.special import expit


def load_glmhmm_data(data_file):
    container = np.load(data_file, allow_pickle=True)
    data = [container[key] for key in container]
    this_hmm_params = data[0]
    lls = data[1]
    return [this_hmm_params, lls]


def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    y = y.astype('int')
    session = data[2]
    return inpt, y, session


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


def load_cv_arr(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    cvbt_folds_model = data[0]
    return cvbt_folds_model


def load_animal_list(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list


def load_reward_data(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    rewarded = data[0]
    return rewarded


def load_correct_incorrect_mat(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    correct_mat = data[0]
    num_trials = data[1]
    return correct_mat, num_trials


def load_rts(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    rt_dta = data[0]
    rt_session = data[1]
    return rt_dta, rt_session


def read_bootstrapped_median(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    median, lower, upper, mean_viol_rate_dist = data[0], data[1], data[2], \
                                                data[3]
    return median, lower, upper, mean_viol_rate_dist


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
        loc_best, :]))[0][0]
    base_path = overall_dir + '/GLM_HMM_K_' + str(K) + '/fold_' + str(
        best_fold)
    key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(best_fold)
    best_iter = best_init_cvbt_dict[key_for_dict]
    raw_file = base_path + '/iter_' + str(
        best_iter) + '/glm_hmm_raw_parameters_itr_' + str(best_iter) + '.npz'
    return raw_file


def create_train_test_trials_for_pred_acc(y, num_folds=5):
    # only select trials that are not violation trials for prediction:
    num_trials = len(np.where(y[:, 0] != -1)[0])
    # Map sessions to folds:
    unshuffled_folds = np.repeat(np.arange(num_folds),
                                 np.ceil(num_trials / num_folds))
    shuffled_folds = npr.permutation(unshuffled_folds)[:num_trials]
    assert len(np.unique(shuffled_folds)
               ) == 5, "require at least one session per fold for each animal!"
    # Look up table of shuffle-folds:
    shuffled_folds = np.array(shuffled_folds, dtype='O')
    trial_fold_lookup_table = np.transpose(
        np.vstack([np.where(y[:, 0] != -1), shuffled_folds]))
    return trial_fold_lookup_table


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
                             "" \
                             "all dta!"
    return nonviolation_idx, mask


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


def calculate_predictive_acc_glm(glm_weights, inpt, y, idx_to_exclude):
    M = inpt.shape[1]
    C = 2
    # Calculate test loglikelihood
    from GLM import glm
    new_glm = glm(M, C)
    # Set parameters to fit parameters:
    new_glm.params = glm_weights
    # time dependent logits:
    prob_right = np.exp(new_glm.calculate_logits(inpt))
    prob_right = prob_right[:, 0, 1]
    # Get the predicted label for each time step:
    predicted_label = np.around(prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(
        y[idx_to_exclude,
          0] == predicted_label[idx_to_exclude]) / len(idx_to_exclude)
    return predictive_acc


def calculate_predictive_acc_lapse_model(lapse_glm_weights, lapse_p,
                                         num_lapse_params, inpt, y,
                                         idx_to_exclude):
    M = inpt.shape[1]
    from LapseModel import lapse_model
    new_lapse_model = lapse_model(M, num_lapse_params)
    if num_lapse_params == 1:
        new_lapse_model.params = [lapse_glm_weights, np.array([lapse_p])]
    else:
        new_lapse_model.params = [lapse_glm_weights, lapse_p]
    prob_right = np.exp(new_lapse_model.calculate_logits(inpt))
    prob_right = prob_right[:, 1]
    # Get the predicted label for each time step:
    predicted_label = np.around(prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(
        y[idx_to_exclude,
          0] == predicted_label[idx_to_exclude]) / len(idx_to_exclude)
    return predictive_acc


def calculate_predictive_accuracy(inputs, datas, train_masks, hmm_params, K,
                                  permutation, transition_alpha, prior_sigma,
                                  y, idx_to_exclude):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    this_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                       transitions="sticky",
                       transition_kwargs=dict(alpha=transition_alpha, kappa=0))
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [
        this_hmm.expected_states(data=data,
                                 input=input,
                                 mask=np.expand_dims(mask, axis=1))[0]
        for data, input, mask in zip(datas, inputs, train_masks)
    ]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, permutation]
    prob_right = [
        np.exp(this_hmm.observations.calculate_logits(input=input))
        for data, input, train_mask in zip(datas, inputs, train_masks)
    ]
    prob_right = np.concatenate(prob_right, axis=0)
    # Now multiply posterior probs and prob_right:
    prob_right = prob_right[:, :, 1]
    # Now multiply posterior probs and prob_right and sum over latent axis:
    final_prob_right = np.sum(np.multiply(posterior_probs, prob_right), axis=1)
    # Get the predicted label for each time step:
    predicted_label = np.around(final_prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(
        y[idx_to_exclude,
          0] == predicted_label[idx_to_exclude]) / len(idx_to_exclude)
    return predictive_acc


def get_marginal_posterior(inputs, datas, masks, hmm_params, K, permutation):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    this_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2),
                       transitions="standard")
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [
        this_hmm.expected_states(data=data,
                                 input=input,
                                 mask=np.expand_dims(mask, axis=1))[0]
        for data, input, mask in zip(datas, inputs, masks)
    ]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, permutation]
    return posterior_probs


def get_prob_right(weight_vectors, inpt, k, pc, wsls):
    # stim vector
    min_val_stim = np.min(inpt[:, 0])
    max_val_stim = np.max(inpt[:, 0])
    stim_vals = np.arange(min_val_stim, max_val_stim, 0.05)
    # create input matrix - cols are stim, pc, wsls, bias
    x = np.array([
        stim_vals,
        np.repeat(pc, len(stim_vals)),
        np.repeat(wsls, len(stim_vals)),
        np.repeat(1, len(stim_vals))
    ]).T
    wx = np.matmul(x, weight_vectors[k][0])
    return stim_vals, expit(wx)


def calculate_correct_ans(y, rewarded):
    # Based on animal's choices and correct response, calculate correct side
    # for each trial (necessary for 0 contrast)
    correct_answer = []
    for i in range(y.shape[0]):
        if rewarded[i, 0] == 1:
            correct_answer.append(y[i, 0])
        else:
            correct_answer.append((y[i, 0] + 1) % 2)
    return correct_answer


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


def get_global_weights(global_directory, K):
    cv_file = global_directory + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(global_directory + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                 global_directory,
                                                 best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    permutation = calculate_state_permutation(hmm_params)
    global_weights = -hmm_params[2][permutation]
    return global_weights


def get_global_trans_mat(global_directory, K):
    cv_file = global_directory + "/cvbt_folds_model.npz"
    cvbt_folds_model = load_cv_arr(cv_file)

    with open(global_directory + "/best_init_cvbt_dict.json", 'r') as f:
        best_init_cvbt_dict = json.load(f)

    # Get the file name corresponding to the best initialization for given K
    # value
    raw_file = get_file_name_for_best_model_fold(cvbt_folds_model, K,
                                                 global_directory,
                                                 best_init_cvbt_dict)
    hmm_params, lls = load_glmhmm_data(raw_file)
    transition_matrix = np.exp(hmm_params[1][0])
    permutation = calculate_state_permutation(hmm_params)
    global_transition_matrix = permute_transition_matrix(
        transition_matrix, permutation)
    return global_transition_matrix


def get_was_correct(this_inpt, this_y):
    '''
    return a vector of size this_y.shape[0] indicating if
    choice was correct on current trial.  Return NA if trial was not "easy"
    trial
    :param this_inpt:
    :param this_y:
    :return:
    '''
    was_correct = np.empty(this_y.shape[0])
    was_correct[:] = np.NaN
    idx_easy = np.where(np.abs(this_inpt[:, 0]) > 0.002)
    correct_side = (np.sign(this_inpt[idx_easy, 0]) + 1) / 2
    was_correct[idx_easy] = (correct_side == this_y[idx_easy, 0]) + 0
    return was_correct, idx_easy


def find_change_points(states_max_posterior):
    '''
    find last trial before change point
    :param states_max_posterior: list of size num_sess; each element is an
    array of size number of trials in session
    :return: list of size num_sess with idx of last trial before a change point
    '''
    num_sess = len(states_max_posterior)
    change_points = []
    for sess in range(num_sess):
        if len(states_max_posterior[sess]) == 90:
            # get difference between consec states
            diffs = np.diff(states_max_posterior[sess])
            # Get locations of all change points
            idx_change_points = np.where(np.abs(diffs) > 0)[0]
            change_points.append(idx_change_points)
    assert len(change_points) == num_sess
    return change_points


def perform_bootstrap_individual_animal(rt_eng_vec,
                                        rt_dis_vec,
                                        data_quantile,
                                        quantile=0.9):
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
    frac_above_true = np.sum(distribution >= data_quantile) / len(distribution)
    return lower, upper, min_val, max_val, frac_above_true

import os

import autograd.numpy as np
import autograd.numpy.random as npr
import sys

from softmax_observations_fit_utils import fit_hmm_observations


def load_data(animal_file):
    container = np.load(animal_file, allow_pickle = True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session

def load_old_ibl_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    y = y.astype('int')
    session = data[3]
    return inpt, y, session

def load_cluster_arr(cluster_arr_file):
    container = np.load(cluster_arr_file, allow_pickle = True)
    data = [container[key] for key in container]
    cluster_arr = data[0]
    return cluster_arr

def load_glm_vectors(glm_vectors_file):
    container = np.load(glm_vectors_file)
    data = [container[key] for key in container]
    loglikelihood_train = data[0]
    recovered_weights = data[1]
    std_dev = data[2]
    return loglikelihood_train, recovered_weights, std_dev

def load_lapse_params(lapse_file):
    container = np.load(lapse_file, allow_pickle=True)
    data = [container[key] for key in container]
    lapse_loglikelihood = data[0]
    lapse_glm_weights = data[1]
    lapse_glm_weights_std = data[2],
    lapse_p = data[3]
    lapse_p_std = data[4]
    return lapse_loglikelihood, lapse_glm_weights, lapse_glm_weights_std, lapse_p, lapse_p_std


def load_global_params(global_params_file):
    container = np.load(global_params_file, allow_pickle = True)
    data = [container[key] for key in container]
    #print(data)
    global_params = data[0]
    return global_params

def load_global_params_old_ibl(global_params_file):
    container = np.load(global_params_file, allow_pickle = True)
    data = [container[key] for key in container]
    global_params = data
    global_params = [global_params[0], [global_params[1]], global_params[2]]
    #print(global_params)
    return global_params

def partition_data_by_session(inpt, y, mask, session):
    '''
    Partition inpt, y, mask by session
    :param inpt: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or not
    :param session: list of size T containing session ids
    :return: list of inpt arrays, data arrays and mask arrays, where the number of elements in list = number of sessions and each array size is number of trials in session
    '''
    inputs = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(indexes)]
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(inpt[idx,:])
        datas.append(y[idx, :])
        masks.append(mask[idx])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks


def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table

def load_animal_list(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list

def run_inference(iter, inputs, datas, train_masks, K, D, M, C, N_em_iters, transition_alpha, prior_sigma, global_fit, params_for_initialization, save_directory):
    npr.seed(iter)
    this_directory = save_directory + '/iter_' + str(iter) + '/'
    if not os.path.exists(this_directory):
        os.makedirs(this_directory)
    fit_hmm_observations(datas, inputs, train_masks, K, D, M, C, N_em_iters, transition_alpha, prior_sigma, global_fit, params_for_initialization, save_title = this_directory + 'glm_hmm_raw_parameters_itr_' + str(iter) + '.npz')
    return None


def launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table, K, D, C, N_em_iters, transition_alpha, prior_sigma, fold, iter, global_fit, init_param_file, overall_dir, ibl_init):
    print("Starting inference with K = " + str(K) + "; Fold = " + str(fold) + "; Iter = " + str(iter))
    sys.stdout.flush()
    sessions_to_keep = session_fold_lookup_table[np.where(session_fold_lookup_table[:, 1] != fold), 0]
    idx_this_fold = [str(sess) in sessions_to_keep for sess in session]
    this_inpt, this_y, this_session, this_mask = inpt[idx_this_fold, :], y[idx_this_fold, :], session[idx_this_fold], mask[idx_this_fold]
    # Only do this so that errors are avoided - these y values will not actually be used for anything
    this_y[np.where(this_y==-1),:] = 1
    inputs, datas, train_masks = partition_data_by_session(this_inpt, this_y, this_mask, this_session)
    # Read in GLM fit if global_fit = True:
    if global_fit == True:
        loglikelihood_train, glm_vectors, glm_std_dev = load_glm_vectors(init_param_file)
        params_for_initialization = glm_vectors
    elif ibl_init == True:
        # if individual fits, initialize each model with the global fit:
        params_for_initialization = load_global_params_old_ibl(init_param_file)
    else:
        params_for_initialization = load_global_params(init_param_file)
    M = this_inpt.shape[1]
    save_directory = overall_dir + '/GLM_HMM_K_' + str(K) + '/' + 'fold_' + str(fold) + '/'
    os.makedirs(save_directory, exist_ok=True)
    run_inference(iter, inputs, datas, train_masks, K, D, M, C, N_em_iters, transition_alpha, prior_sigma, global_fit,
                  params_for_initialization, save_directory)

def create_violation_mask(violation_idx, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1 = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in violation_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonviolation_idx) + len(violation_idx) == T, "violation and non-violation idx do not include all dta!"
    return nonviolation_idx, mask

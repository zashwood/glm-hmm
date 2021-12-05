import autograd.numpy as np
from fit_glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, load_data, create_violation_mask, launch_glm_hmm_job
import sys

D = 1    # data (observations) dimension
C = 2    # number of output types/categories
N_em_iters = 300  # number of EM iterations


if __name__ == '__main__':
    z = int(sys.argv[1])
    data_dir = 'data_for_cluster/'
    num_folds = 5

    # Load external files:
    cluster_arr_file = data_dir + 'cluster_job_arr.npz'
    # Load cluster array job parameters:
    cluster_arr = load_cluster_arr(cluster_arr_file)
    [K, fold, iter] = cluster_arr[z]

    animal_file = data_dir + 'all_animals_concat.npz'
    session_fold_lookup_table = load_session_fold_lookup(data_dir + 'all_animals_concat_session_fold_lookup.npz')

    global_fit = True

    inpt, y, session = load_data(animal_file)
    overall_dir = 'results/ibl_global_fit/' #+ 'covar_set_' + str(covar_set) + '/'
    inpt = inpt[:, [0, 1, 2]]

    y = y.astype('int')
    # Identify violations for exclusion:
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])

    init_param_file = overall_dir + '/GLM/fold_' + str(fold) + '/variables_of_interest_iter_0.npz'

    # perform mle => set transition_alpha to 1
    transition_alpha = 1
    prior_sigma = 100
    launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table, K, D, C, N_em_iters, transition_alpha,
                       prior_sigma, fold, iter, global_fit, init_param_file, overall_dir, ibl_init=False)
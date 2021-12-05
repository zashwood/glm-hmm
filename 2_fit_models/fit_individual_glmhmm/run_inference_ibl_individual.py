import autograd.numpy as np
from fit_global_glmhmm.fit_glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, load_data, load_animal_list, create_violation_mask, launch_glm_hmm_job
import sys

D = 1    # data (observations) dimension
C = 2    # number of output types/categories
N_em_iters = 300  # number of EM iterations


if __name__ == '__main__':
    z = int(sys.argv[1])
    data_dir = 'data/ibl/data_for_cluster/'
    num_folds = 5

    # Load external files:
    cluster_arr_file = data_dir + 'data_by_animal/cluster_job_arr.npz'
    # Load cluster array job parameters:
    cluster_arr = load_cluster_arr(cluster_arr_file)
    [prior_sigma, transition_alpha, K, fold, iter] = cluster_arr[z]

    iter = int(iter)
    fold = int(fold)
    K=int(K)

    animal_list = load_animal_list(data_dir + 'data_by_animal/animal_list.npz')

    covar_set = 2

    for i, animal in enumerate(animal_list):
        print(animal)
        animal_file = data_dir + 'data_by_animal/' +  animal + '_processed.npz'
        session_fold_lookup_table = load_session_fold_lookup(data_dir + 'data_by_animal/' + animal + '_session_fold_lookup.npz')

        global_fit = False

        inpt, y, session = load_data(animal_file)
        overall_dir = 'results/ibl_individual_fit/covar_set_' + str(covar_set) +  '/prior_sigma_' + str(prior_sigma) +'_transition_alpha_' + str(transition_alpha) + '/' + animal + '/'
        # Subset to relevant covariates for covar set of interest:
        if covar_set == 0:
            inpt = inpt[:, [0]]
            print(inpt.shape)
        elif covar_set == 1:
            inpt = inpt[:, [0, 1]]
            print(inpt.shape)
        elif covar_set == 2:
            inpt = inpt[:, [0, 1, 2]]
            print(inpt.shape)


        y = y.astype('int')
        # Identify violations for exclusion:
        violation_idx = np.where(y == -1)[0]
        nonviolation_idx, mask = create_violation_mask(violation_idx, inpt.shape[0])


        init_param_file = data_dir + 'best_params/best_params_K_' + str(K) + '_covar_set_' + str(covar_set) + '.npz'

        launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table, K, D, C, N_em_iters, transition_alpha,
                           prior_sigma, fold, iter, global_fit, init_param_file, overall_dir, ibl_init=False)
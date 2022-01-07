#  In order to facilitate parallelization of jobs, create a job array that
#  can be used on e.g. a cluster
import numpy as np

prior_sigma = [2]
transition_alpha = [2]
K_vals = [2, 3, 4, 5]
num_folds = 5
N_initializations = 2

if __name__ == '__main__':
    cluster_job_arr = []
    for K in K_vals:
        for i in range(num_folds):
            for j in range(N_initializations):
                for sigma in prior_sigma:
                    for alpha in transition_alpha:
                        cluster_job_arr.append([sigma, alpha, K, i, j])
    np.savez('../../data/ibl/data_for_cluster/data_by_animal/cluster_job_arr'
             '.npz',
             cluster_job_arr)
    print(len(cluster_job_arr))

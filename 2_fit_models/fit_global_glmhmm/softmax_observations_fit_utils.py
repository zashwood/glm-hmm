import autograd.numpy as np
import ssm
from autograd.scipy.special import logsumexp
import sys

def create_hmm_simulated_data(K, D, C, inpt, glm_vectors):
    T =inpt.shape[0]
    M = inpt.shape[1]
    this_hmm = HMM(K, D, M, 
        observations="softmax", observation_kwargs=dict(C=C),
        transitions="standard")
    glm_vectors_repeated = np.tile(glm_vectors, (K,1,1))
    glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(0, 0.1, glm_vectors_repeated.shape)
    this_hmm.observations.params = glm_vectors_with_noise
    z, y = this_hmm.sample(T, input = inpt)
    true_ll = this_hmm.log_probability([y], inputs =[inpt])
    print("True ll = " + str(true_ll))
    return z, y, true_ll, this_hmm.params



def fit_hmm_observations(datas, inputs, train_masks, K, D, M, C, N_em_iters, transition_alpha, prior_sigma, global_fit, params_for_initialization, save_title):
    '''
    Instantiate and fit GLM-HMM model
    :param datas:
    :param inputs:
    :param train_masks:
    :param K:
    :param D:
    :param M:
    :param C:
    :param N_em_iters:
    :param global_fit:
    :param glm_vectors:
    :param save_title:
    :return:
    '''
    if global_fit == True:
        # Prior variables
        # Choice of prior
        this_hmm = ssm.HMM(K, D, M,
                           observations="softmax", observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                           transitions="sticky", transition_kwargs=dict(alpha=transition_alpha, kappa=0))
        print(this_hmm.params)
        # Initialize observation weights as GLM weights with some noise:
        glm_vectors_repeated = np.tile(params_for_initialization, (K, 1, 1))
        glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(0, 0.2, glm_vectors_repeated.shape)
        this_hmm.observations.params = glm_vectors_with_noise
        print(this_hmm.params)
        sys.stdout.flush()
    else:
        # Choice of prior
        this_hmm = ssm.HMM(K, D, M,
                           observations="softmax", observation_kwargs=dict(C=C, prior_sigma=prior_sigma),
                           transitions="sticky", transition_kwargs=dict(alpha=transition_alpha, kappa=0))
        print(this_hmm.params)
        # Initialize HMM-GLM with global parameters:
        this_hmm.params = params_for_initialization
        print(this_hmm.params)
        # Get log_prior of transitions:
        print(this_hmm.log_prior())
        print(this_hmm.transitions.log_prior())
    print("=== fitting HMM ========")
    sys.stdout.flush()
    # Fit this HMM and calculate marginal likelihood
    lls = this_hmm.fit(datas, inputs=inputs, train_masks=train_masks, method="em", num_iters=N_em_iters, tolerance=10**-4)
    # Save raw parameters of HMM, as well as loglikelihood and accuracy calculated during training
    np.savez(save_title, this_hmm.params, lls)
    return None





def permute_z_inf(z_inf, permutation):
    # Now modify inferred_z so that labeling of latents matches that of true z:
    perm_dict = dict(zip(permutation, range(len(permutation))))
    inferred_z = np.array([perm_dict[x] for x in z_inf])
    return inferred_z       

# Calculate prediction accuracy of GLM-HMM
def calculate_prediction_accuracy(y, inpt, this_hmm):
    # Calculate most probable observation class at each time point:
    time_dependent_logits = this_hmm.observations.calculate_logits(inpt)
    # Now marginalize over the latent dimension:
    time_dependent_class_log_probs = logsumexp(time_dependent_logits, axis = 1)
    assert time_dependent_class_log_probs.shape == (inpt.shape[0], time_dependent_logits.shape[2]), "wrong shape for time_dependent_class_log_probs"
    # Now find the location of the max along the C dimension
    predicted_class_labels = np.argmax(time_dependent_class_log_probs, axis = 1)
    # Calculate where y and predicted_class_labels line up:
    predictive_acc = np.sum(y[:,0] == predicted_class_labels)/y.shape[0]
    return predictive_acc




# Append column of zeros to weights matrix in appropriate location
def append_zeros(weights):
    weights_tranpose = np.transpose(weights, (1,0,2))
    weights = np.transpose(np.vstack([weights_tranpose, np.zeros((1,weights_tranpose.shape[1], weights_tranpose.shape[2]))]), (1,0,2))
    return weights


# Reshape hessian and calculate its inverse
def calculate_std_hmm(hessian, permutation, M, K, C):
    # Reshape hessian
    hessian = np.reshape(hessian, ((K*(C-1)*(M+1)),(K*(C-1)*(M+1))))
    # Calculate inverse of Hessian (this is what we will actually use to calculate variance)
    inv_hessian = hessian ## TODO: remove this when we have a Hessian that is not the identity
    # Take diagonal elements and calculate square root
    std_dev = np.sqrt(np.diag(inv_hessian))
    # Undo permutation
    unflattened_std_dev = np.reshape(std_dev, (K, C-1, M+1))
    # Append zeros
    unflattened_std_dev = append_zeros(unflattened_std_dev)
    # Now undo permutation:
    unflattened_std_dev = unflattened_std_dev[permutation]
    # Now reflatten std_dev (for ease of plotting)
    flattened_std_dev = unflattened_std_dev.flatten()
    return flattened_std_dev



















# Prepare IBL data from publicly available dataset:

import numpy as np
from oneibl.onelight import ONE
from sklearn import preprocessing
from numpy import linalg as LA
import numpy.random as npr
from scipy.stats import bernoulli
import json
from collections import defaultdict
from oneibl.onelight import ONE
import os
npr.seed(65)

def get_all_unnormalized_data_this_session(eid):
    '''
    read in raw behavioral data for session corresponding to eid; process this to create design matrix for GLM-HMM
    :param eid (string): file path indicating where data for particular eid is stored
    :return:
    animal (string): name of animal for this particular session
    unnormalized_inpt (array): array of size Tx3 where the first column is stimulus presented at trial t, second column is
    choice on previous trial and third column is product of reward and choice on previous trial
    y (array): array of size Tx1 indicating animal's choice on current trial (1 is Rightward, 0 is Leftward, -1 is violation)
    session (list): list of size T indicating session_id for trial
    rewarded (array): array of size Tx1 indicating if current trial was rewarded
    '''
    # Load raw data
    animal, session_id, stim_left, stim_right, rewarded, choice, bias_probs = get_raw_data(eid)
    # Subset choice and design_mat to 50-50 entries:
    trials_to_study = np.where(bias_probs == 0.5)[0]
    # Create design mat = matrix of size T x 3, with entries for stim/past choice/wsls
    unnormalized_inpt = create_design_mat(choice[trials_to_study], stim_left[trials_to_study], stim_right[trials_to_study], rewarded[trials_to_study])
    y = np.expand_dims(remap_choice_vals(choice[trials_to_study]), axis =1) #expand dimensions for output vector (required for ssm package)
    session = [session_id for i in range(y.shape[0])] #list of size T repeating the session identifier for each trial
    rewarded = np.expand_dims(rewarded[trials_to_study], axis =1) #matrix of size Tx1 indicating if trial resulted in reward
    return animal, unnormalized_inpt, y, session, rewarded


def get_raw_data(eid):
    '''
    use IBL's "ONE" package to read in raw behavioral data for session of interest (indicated by eid)
    :param eid: eid (string): file path indicating where data for particular eid is stored
    :return: various raw behavioral variables of size T, number of trials in session
    '''
    one = ONE()  # use "ONE" package to identify all individual sessions in the downloaded data
    # get session id:
    raw_session_id = eid.split('Subjects/')[1]
    raw_session_id = eid.split('Subjects/')[1]
    # Get animal:
    animal = raw_session_id.split('/')[0]
    # replace '/' with dash in session ID
    session_id = raw_session_id.replace('/', '-')
    # Get choice data, stim data and rewarded/not rewarded:
    choice = one.load_dataset(eid, '_ibl_trials.choice')
    stim_left = one.load_dataset(eid, '_ibl_trials.contrastLeft')
    stim_right = one.load_dataset(eid, '_ibl_trials.contrastRight')
    rewarded = one.load_dataset(eid, '_ibl_trials.feedbackType')
    bias_probs = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
    return animal, session_id, stim_left, stim_right, rewarded, choice, bias_probs


def create_stim_vector(stim_left, stim_right):
    '''
    take vectors of stim_left and stim_right values and combine to create single stimulus value (stim_right-stim_left)
    for each trial
    :param stim_left:
    :param stim_right:
    :return:
    '''
    # want stim_right - stim_left
    # Replace NaNs with 0:
    stim_left = np.nan_to_num(stim_left, nan=0)
    stim_right = np.nan_to_num(stim_right, nan=0)
    # now get 1D stim
    signed_contrast = stim_right - stim_left
    return signed_contrast


def create_previous_choice_vector(choice):
    ''' choice: choice vector of size T

        previous_choice : vector of size T with previous choice made by animal - output is in {0, 1}, where 0 corresponds to a previous left choice; 1 corresponds to right.
        If the previous choice was a violation, replace this with the choice on the previous trial that was not a violation.

        locs_mapping: array of size (~num_viols)x2, where the entry in column 1 is the location in the previous choice vector that was a remapping due to a violation and the
        entry in column 2 is the location in the previous choice vector that this location was remapped to

    '''
    previous_choice = np.hstack([np.array(choice[0]), choice])[:-1]
    locs_to_update = np.where(previous_choice == -1)[0]
    locs_with_choice = np.where(previous_choice != -1)[0]
    loc_first_choice = locs_with_choice[0]
    locs_mapping = np.zeros((len(locs_to_update) - loc_first_choice, 2), dtype='int')

    for i, loc in enumerate(locs_to_update):
        if loc < loc_first_choice:
            # since no previous choice, bernoulli sample: (not output of bernoulli rvs is in {1, 2})
            previous_choice[loc] = bernoulli.rvs(0.5, 1) - 1
        else:
            # find nearest loc that has a previous choice value that is not -1, and that is earlier than current trial
            # print("loc to match " + str(loc))
            potential_matches = locs_with_choice[np.where(locs_with_choice < loc)]
            # print("current loc = " + str(loc))
            absolute_val_diffs = np.abs(loc - potential_matches)
            # print(absolute_val_diffs)
            absolute_val_diffs_ind = absolute_val_diffs.argmin()
            nearest_loc = potential_matches[absolute_val_diffs_ind]
            # print("matched loc " + str(nearest_loc))
            # print("nearest loc = " + str(nearest_loc))
            locs_mapping[i - loc_first_choice, 0] = int(loc)
            locs_mapping[i - loc_first_choice, 1] = int(nearest_loc)
            previous_choice[loc] = previous_choice[nearest_loc]
    assert len(np.unique(previous_choice)) <= 2, "previous choice should be in {0, 1}; " + str(np.unique(previous_choice))
    assert len(locs_to_update) < 11, "should not include session if more than 11 viols: num viols = " + str(len(locs_to_update)) # mismatch here because sometimes a viol will be added to first position
    return previous_choice, locs_mapping


def create_wsls_covariate(previous_choice, success, locs_mapping):
    '''
    inputs:
    success: vector of size T, entries are in {-1, 1} and 0 corresponds to failure, 1 corresponds to success
    previous_choice: vector of size T, entries are in {0, 1} and 0 corresponds to left choice, 1 corresponds to right choice
    locs_mapping: location remapping dictionary due to violations

    output:
    wsls: vector of size T, entries are in {-1, 1}.  1 corresponds to previous choice = right and success OR previous choice = left and failure; -1 corresponds to
    previous choice = left and success OR previous choice = right and failure

    '''
    # remap previous choice vals to {-1, 1}
    remapped_previous_choice = 2 * previous_choice - 1
    previous_reward = np.hstack([np.array(success[0]), success])[:-1]
    # Now need to go through and update previous reward to correspond to same trial as previous choice:
    for i, loc in enumerate(locs_mapping[:, 0]):
        nearest_loc = locs_mapping[i, 1]
        previous_reward[loc] = previous_reward[nearest_loc]
    wsls = previous_reward * remapped_previous_choice
    assert len(np.unique(wsls)) == 2, "wsls should be in {-1, 1}"
    return wsls


def remap_choice_vals(choice):
    # raw choice vector has CW = 1 (correct response for stim on left), CCW = -1 (correct response for stim on right) and viol = 0.  Let's remap so that CW = 0, CCw = 1, and viol = -1
    choice_mapping = {1:0, -1:1, 0:-1}
    new_choice_vector = [choice_mapping[old_choice] for old_choice in choice]
    return new_choice_vector


def create_design_mat(choice, stim_left, stim_right, rewarded):
    '''
    create unnormalized_inpt: with first column = stim_right - stim_left, second column as past choice, third column as WSLS
    :param choice (vector): vector of size T indicating choice on each trial
    :param stim_left (vector): vector of size T indicating leftward stimulus on each trial
    :param stim_right (vector): vector of size T indicating rightward stimulus on each trial
    :param rewarded (vector): vector of size T indicating if animal was rewarded on each trial
    :return:
    '''
    stim = create_stim_vector(stim_left, stim_right)
    T = len(stim)
    design_mat = np.zeros((T, 3))
    design_mat[:, 0] = stim
    # make choice vector so that correct response for stim>0 is choice =1 and is 0 for stim <0 (viol is mapped to -1)
    choice = remap_choice_vals(choice)
    # create past choice vector:
    previous_choice, locs_mapping = create_previous_choice_vector(choice)
    # create wsls vector:
    wsls = create_wsls_covariate(previous_choice, rewarded, locs_mapping)
    # map previous choice to {-1,1}
    design_mat[:,1] = 2 * previous_choice - 1
    design_mat[:,2] = wsls
    return design_mat


# def load_animal_list(file):
#     container = np.load(file, allow_pickle=True)
#     data = [container[key] for key in container]
#     animal_list = data[0]
#     return animal_list
#
# def load_animal_eid_dict(file):
#     with open(file, 'r') as f:
#         animal_eid_dict = json.load(f)
#     return animal_eid_dict

def create_train_test_sessions(session, num_folds = 5):
    '''
    create cross-validation train-test splits. produce a
    :param session:
    :param num_folds:
    :return:
    '''
    # create a session-fold lookup table
    num_sessions = len(np.unique(session))
    # Map sessions to folds:
    unshuffled_folds = np.repeat(np.arange(num_folds), np.ceil(num_sessions/num_folds))
    shuffled_folds = npr.permutation(unshuffled_folds)[:num_sessions]
    assert len(np.unique(shuffled_folds)) == 5, "require at least one session per fold for each animal!"
    # Look up table of shuffle-folds:
    sess_id = np.array(np.unique(session), dtype='str')
    shuffled_folds = np.array(shuffled_folds, dtype = 'O')
    session_fold_lookup_table = np.transpose(np.vstack([sess_id, shuffled_folds]))
    return session_fold_lookup_table


def identify_bias_block_sessions(ibl_data_path):
    '''
    identify "bias block" sessions (see description of IBL task) for each IBL animal in downloaded dataset
    :return:
    animal_list (list): list of all animals in downloaded dataset with bias block sessions
    animal_eid_dict (dict): dictionary with animal names (keys) and list of file paths to all bias block sessions for these animals
    '''
    def _get_animal_name(eid):
        '''
        helper function for identify_bias_block_sessions.  Takes returned file path and extracts animal name
        :param eid (string): file path for session of bias block data for a particular animal
        :return: animal(string): string indicating name of animal; extracted from file path
        '''
        # get session id:
        raw_session_id = eid.split('Subjects/')[1]
        # Get animal:
        animal = raw_session_id.split('/')[0]
        return animal
    one = ONE()  #use "ONE" package to identify all individual sessions in the downloaded data
    current_cwd = os.getcwd()
    os.chdir(ibl_data_path) # ONE search function requires that we change directory to location of saved data to use search function
    eids = one.search(['_ibl_trials.*'])
    animal_list = []
    animal_eid_dict = defaultdict(list)
    for eid in eids:
        bias_probs = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
        comparison = np.unique(bias_probs)==np.array([0.2, 0.5, 0.8]) #confirm that session is "bias block" session
        if isinstance(comparison, np.ndarray): # update def of comparison to single True/False
            comparison = comparison.all()
        if comparison == True:
            animal = _get_animal_name(eid)
            if animal not in animal_list:
                animal_list.append(animal)
            animal_eid_dict[animal].append(eid)
    os.chdir(current_cwd) # switch directory back
    return animal_list, animal_eid_dict







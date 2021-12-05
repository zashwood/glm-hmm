# Prepare IBL data from publicly available dataset:

import numpy as np
from oneibl.onelight import ONE
from sklearn import preprocessing
from numpy import linalg as LA
import numpy.random as npr
from scipy.stats import bernoulli
import json
from collections import defaultdict
npr.seed(65)


def get_raw_data(eid):
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
    # Create unnormalized_inpt: with first column = stim_right - stim_left, second column as past choice, third column as WSLS
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


def get_all_unnormalized_data_this_session(eid):
    # Load raw data
    animal, session_id, stim_left, stim_right, rewarded, choice, bias_probs = get_raw_data(eid)
    # Subset choice and design_mat to 50-50 entries:
    trials_to_study = np.where(bias_probs == 0.5)[0]
    num_viols_50 = len(np.where(choice[trials_to_study]==0)[0])
    if num_viols_50 <10:
        # Create design mat = matrix of size T x 3, with entries for stim/past choice/wsls
        unnormalized_inpt = create_design_mat(choice[trials_to_study], stim_left[trials_to_study], stim_right[trials_to_study], rewarded[trials_to_study])
        y = np.expand_dims(remap_choice_vals(choice[trials_to_study]), axis =1)
        session = [session_id for i in range(y.shape[0])]
        rewarded = np.expand_dims(rewarded[trials_to_study], axis =1)
    else:
        unnormalized_inpt = np.zeros((90, 3))
        y = np.zeros((90, 1))
        session = []
        rewarded = np.zeros((90, 1))
    return animal, unnormalized_inpt, y, session, num_viols_50, rewarded

def calculate_condition_number(inpt):
    full_inpt = np.hstack((inpt, np.ones((inpt.shape[0], 1))))
    condition_number = LA.cond(full_inpt)
    return condition_number

def load_animal_list(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list

def load_animal_eid_dict(file):
    with open(file, 'r') as f:
        animal_eid_dict = json.load(f)
    return animal_eid_dict

def create_train_test_sessions(session, num_folds = 5):
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



if __name__ == '__main__':
    data_dir = 'data_for_cluster/'
    one = ONE()

    # Load animal list:
    animal_list = load_animal_list('partially_processed/animal_list.npz')
    #animal_list = ['CSHL_008']
    # Load animal-eid dict (keys are animals and vals are list of eids for biased block sessions)
    animal_eid_dict = load_animal_eid_dict('partially_processed/animal_eid_dict.json')

    # Require that each animal has at least 30 sessions (=2700 trials) of data:
    req_num_sessions = 30 #30*90 = 2700
    for animal in animal_list:
        num_sessions = len(animal_eid_dict[animal])
        if num_sessions < req_num_sessions:
            animal_list = np.delete(animal_list, np.where(animal_list==animal))

    # Identify idx in master array where each animal's data starts and ends:
    animal_start_idx = {}
    animal_end_idx = {}

    final_animal_eid_dict = defaultdict(list)
    # WORKHORSE: iterate through each animal and each animal's set of eids; obtain unnormalized data.  Write out each animal's data and then also write to master array
    for z, animal in enumerate(animal_list):
        sess_counter = 0
        for eid in animal_eid_dict[animal]:
            animal, unnormalized_inpt, y, session, num_viols_50, rewarded = get_all_unnormalized_data_this_session(eid)
            if num_viols_50 < 10: # only include session if number of viols in 50-50 block is less than 10
                if sess_counter == 0:
                    animal_unnormalized_inpt = np.copy(unnormalized_inpt)
                    animal_y = np.copy(y)
                    animal_session = session
                    animal_rewarded = np.copy(rewarded)
                else:
                    animal_unnormalized_inpt = np.vstack((animal_unnormalized_inpt, unnormalized_inpt))
                    animal_y = np.vstack((animal_y, y))
                    animal_session = np.concatenate((animal_session, session))
                    animal_rewarded = np.vstack((animal_rewarded, rewarded))
                sess_counter += 1
                final_animal_eid_dict[animal].append(eid)
        # Write out animal's unnormalized data matrix:
        np.savez(data_dir + 'data_by_animal/' + animal + '_unnormalized.npz', animal_unnormalized_inpt, animal_y,
                 animal_session)
        animal_session_fold_lookup = create_train_test_sessions(animal_session, 5)
        np.savez(data_dir + 'data_by_animal/' + animal + "_session_fold_lookup" + ".npz",
                 animal_session_fold_lookup)
        np.savez(data_dir + 'data_by_animal/' + animal + '_rewarded.npz', animal_rewarded)
        assert animal_rewarded.shape[0] == animal_y.shape[0]
        # Now create or append data to master array across all animals:
        if z == 0:
            master_inpt = np.copy(animal_unnormalized_inpt)
            animal_start_idx[animal] = 0
            animal_end_idx[animal] = master_inpt.shape[0] - 1
            master_y = np.copy(animal_y)
            master_session = animal_session
            master_session_fold_lookup_table = animal_session_fold_lookup
            master_rewarded = np.copy(animal_rewarded)
        else:
            animal_start_idx[animal] = master_inpt.shape[0]
            master_inpt = np.vstack((master_inpt, animal_unnormalized_inpt))
            animal_end_idx[animal] = master_inpt.shape[0] - 1
            master_y = np.vstack((master_y, animal_y))
            master_session = np.concatenate((master_session, animal_session))
            master_session_fold_lookup_table = np.vstack(
                (master_session_fold_lookup_table, animal_session_fold_lookup))
            master_rewarded = np.vstack((master_rewarded, animal_rewarded))
    # Write out data from across animals
    assert np.shape(master_inpt)[0] == np.shape(master_y)[0], "inpt and y not same length"
    assert np.shape(master_rewarded)[0] == np.shape(master_y)[0], "rewarded and y not same length"
    assert len(np.unique(master_session)) == np.shape(master_session_fold_lookup_table)[
        0], "number of unique sessions and session fold lookup don't match"
    normalized_inpt = np.copy(master_inpt)
    normalized_inpt[:,0] = preprocessing.scale(normalized_inpt[:,0])
    calculate_condition_number(preprocessing.scale(normalized_inpt))
    np.savez(data_dir + 'all_animals_concat' + '.npz', normalized_inpt, master_y, master_session)
    np.savez(data_dir + 'all_animals_concat_unnormalized' + '.npz', master_inpt, master_y, master_session)
    np.savez(data_dir + 'all_animals_concat_session_fold_lookup' + '.npz', master_session_fold_lookup_table)
    np.savez(data_dir + 'all_animals_concat_rewarded' + '.npz', master_rewarded)
    np.savez(data_dir + 'data_by_animal/' + 'animal_list.npz', animal_list)

    json = json.dumps(final_animal_eid_dict)
    f = open(data_dir+"final_animal_eid_dict.json", "w")
    f.write(json)
    f.close()

    # Now write out normalized data (when normalized across all animals) for each animal:
    counter = 0
    for animal in animal_start_idx.keys():
        start_idx = animal_start_idx[animal]
        end_idx = animal_end_idx[animal]
        inpt = normalized_inpt[range(start_idx, end_idx + 1)]
        y = master_y[range(start_idx, end_idx + 1)]
        session = master_session[range(start_idx, end_idx + 1)]
        counter += inpt.shape[0]
        np.savez(data_dir + 'data_by_animal/' + animal + '_processed.npz', inpt, y,
                 session)

    assert counter == master_inpt.shape[0]




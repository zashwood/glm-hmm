# Continue preprocessing of IBL dataset and create design matrix for GLM-HMM
import numpy as np
from sklearn import preprocessing
import numpy.random as npr
import os
import json
from collections import defaultdict
from preprocessing_utils import load_animal_list, load_animal_eid_dict, \
    get_all_unnormalized_data_this_session, create_train_test_sessions

npr.seed(65)

if __name__ == '__main__':
    data_dir = '../../data/ibl/'
    # Create directories for saving data:
    processed_ibl_data_path = data_dir + "data_for_cluster/"
    if not os.path.exists(processed_ibl_data_path):
        os.makedirs(processed_ibl_data_path)
    # Also create a subdirectory for storing each individual animal's data:
    if not os.path.exists(processed_ibl_data_path + "data_by_animal/"):
        os.makedirs(processed_ibl_data_path + "data_by_animal/")

    # Load animal list/results of partial processing:
    animal_list = load_animal_list(
        data_dir + 'partially_processed/animal_list.npz')
    animal_eid_dict = load_animal_eid_dict(
        data_dir + 'partially_processed/animal_eid_dict.json')

    # Require that each animal has at least 30 sessions (=2700 trials) of data:
    req_num_sessions = 30  # 30*90 = 2700
    for animal in animal_list:
        num_sessions = len(animal_eid_dict[animal])
        if num_sessions < req_num_sessions:
            animal_list = np.delete(animal_list,
                                    np.where(animal_list == animal))
    # Identify idx in master array where each animal's data starts and ends:
    animal_start_idx = {}
    animal_end_idx = {}

    final_animal_eid_dict = defaultdict(list)
    # WORKHORSE: iterate through each animal and each animal's set of eids;
    # obtain unnormalized data.  Write out each animal's data and then also
    # write to master array
    for z, animal in enumerate(animal_list):
        sess_counter = 0
        for eid in animal_eid_dict[animal]:
            animal, unnormalized_inpt, y, session, num_viols_50, rewarded = \
                get_all_unnormalized_data_this_session(
                    eid)
            if num_viols_50 < 10:  # only include session if number of viols
                # in 50-50 block is less than 10
                if sess_counter == 0:
                    animal_unnormalized_inpt = np.copy(unnormalized_inpt)
                    animal_y = np.copy(y)
                    animal_session = session
                    animal_rewarded = np.copy(rewarded)
                else:
                    animal_unnormalized_inpt = np.vstack(
                        (animal_unnormalized_inpt, unnormalized_inpt))
                    animal_y = np.vstack((animal_y, y))
                    animal_session = np.concatenate((animal_session, session))
                    animal_rewarded = np.vstack((animal_rewarded, rewarded))
                sess_counter += 1
                final_animal_eid_dict[animal].append(eid)
        # Write out animal's unnormalized data matrix:
        np.savez(
            processed_ibl_data_path + 'data_by_animal/' + animal +
            '_unnormalized.npz',
            animal_unnormalized_inpt, animal_y,
            animal_session)
        animal_session_fold_lookup = create_train_test_sessions(animal_session,
                                                                5)
        np.savez(
            processed_ibl_data_path + 'data_by_animal/' + animal +
            "_session_fold_lookup" +
            ".npz",
            animal_session_fold_lookup)
        np.savez(
            processed_ibl_data_path + 'data_by_animal/' + animal +
            '_rewarded.npz',
            animal_rewarded)
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
    assert np.shape(master_inpt)[0] == np.shape(master_y)[
        0], "inpt and y not same length"
    assert np.shape(master_rewarded)[0] == np.shape(master_y)[
        0], "rewarded and y not same length"
    assert len(np.unique(master_session)) == \
           np.shape(master_session_fold_lookup_table)[
               0], "number of unique sessions and session fold lookup don't " \
                   "match"
    assert len(master_inpt) == 181530, "design matrix for all IBL animals " \
                                       "should have shape (181530, 3)"
    assert len(animal_list) == 37, "37 animals were studied in Ashwood et " \
                                   "al. (2020)"
    normalized_inpt = np.copy(master_inpt)
    normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0])
    np.savez(processed_ibl_data_path + 'all_animals_concat' + '.npz',
             normalized_inpt,
             master_y, master_session)
    np.savez(
        processed_ibl_data_path + 'all_animals_concat_unnormalized' + '.npz',
        master_inpt, master_y, master_session)
    np.savez(
        processed_ibl_data_path + 'all_animals_concat_session_fold_lookup' +
        '.npz',
        master_session_fold_lookup_table)
    np.savez(processed_ibl_data_path + 'all_animals_concat_rewarded' + '.npz',
             master_rewarded)
    np.savez(processed_ibl_data_path + 'data_by_animal/' + 'animal_list.npz',
             animal_list)

    json = json.dumps(final_animal_eid_dict)
    f = open(processed_ibl_data_path + "final_animal_eid_dict.json", "w")
    f.write(json)
    f.close()

    # Now write out normalized data (when normalized across all animals) for
    # each animal:
    counter = 0
    for animal in animal_start_idx.keys():
        start_idx = animal_start_idx[animal]
        end_idx = animal_end_idx[animal]
        inpt = normalized_inpt[range(start_idx, end_idx + 1)]
        y = master_y[range(start_idx, end_idx + 1)]
        session = master_session[range(start_idx, end_idx + 1)]
        counter += inpt.shape[0]
        np.savez(processed_ibl_data_path + 'data_by_animal/' + animal + '_processed.npz',
                 inpt, y,
                 session)

    assert counter == master_inpt.shape[0]

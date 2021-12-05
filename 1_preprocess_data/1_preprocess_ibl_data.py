# script to download and preprocess IBL data
import numpy as np
import numpy.random as npr
import wget
from zipfile import ZipFile
import os
from collections import defaultdict
import json
from preprocessing_utils import identify_bias_block_sessions, get_all_unnormalized_data_this_session
npr.seed(65)

if __name__ == '__main__':
    # create save path for IBL data
    ibl_data_path = "../data/ibl/"
    #download IBL data
    url = 'https://ndownloader.figshare.com/files/21623715'
    wget.download(url, ibl_data_path)
    #now unzip downloaded data:
    with ZipFile(ibl_data_path + "ibl-behavior-data-Dec2019.zip", 'r') as zipObj:
        #extract all the contents of zip file in ibl_data_path
        zipObj.extractall(ibl_data_path)
    # now create directory for storing processed data:
    processed_ibl_data_path = ibl_data_path + "data_for_cluster/"
    if not os.path.exists(processed_ibl_data_path):
        os.makedirs(processed_ibl_data_path)
    # Also create a subdirectory for storing each individual animal's data:
    if not os.path.exists(processed_ibl_data_path + "data_by_animal/"):
        os.makedirs(processed_ibl_data_path + "data_by_animal/")
    # identify sessions of interest: we want to subset to the first 90 trials in sessions that occur once the animal
    # has learned the task:
    animal_list, animal_eid_dict = identify_bias_block_sessions(ibl_data_path)
    # now subset further to examine only animals with at least req_num_sessions such sessions of "bias blocks" data:
    updated_animal_list = []
    req_num_sessions = 30
    for animal in animal_list:
        num_sessions = len(animal_eid_dict[animal])
        if num_sessions >= req_num_sessions:
           updated_animal_list.append(animal)

    # Identify idx in master array where each animal's data starts and ends:
    animal_start_idx = {}
    animal_end_idx = {}
    for z, animal in enumerate(updated_animal_list):
        sess_counter = 0
        for eid in animal_eid_dict[animal]:
            animal, unnormalized_inpt, y, session, rewarded = get_all_unnormalized_data_this_session(eid)
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
    np.savez(processed_ibl_data_path + 'data_by_animal/' + animal + '_unnormalized.npz', animal_unnormalized_inpt, animal_y,
             animal_session)
    animal_session_fold_lookup = create_train_test_sessions(animal_session, 5)
    np.savez(processed_ibl_data_path + 'data_by_animal/' + animal + "_session_fold_lookup" + ".npz",
             animal_session_fold_lookup)
    np.savez(processed_ibl_data_path + 'data_by_animal/' + animal + '_rewarded.npz', animal_rewarded)
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
normalized_inpt[:, 0] = preprocessing.scale(normalized_inpt[:, 0])
calculate_condition_number(preprocessing.scale(normalized_inpt))
np.savez(processed_ibl_data_path + 'all_animals_concat' + '.npz', normalized_inpt, master_y, master_session)
np.savez(processed_ibl_data_path + 'all_animals_concat_unnormalized' + '.npz', master_inpt, master_y, master_session)
np.savez(processed_ibl_data_path + 'all_animals_concat_session_fold_lookup' + '.npz', master_session_fold_lookup_table)
np.savez(processed_ibl_data_path + 'all_animals_concat_rewarded' + '.npz', master_rewarded)
np.savez(processed_ibl_data_path + 'data_by_animal/' + 'animal_list.npz', animal_list)

json = json.dumps(final_animal_eid_dict)
f = open(processed_ibl_data_path + "final_animal_eid_dict.json", "w")
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
    np.savez(processed_ibl_data_path + 'data_by_animal/' + animal + '_processed.npz', inpt, y,
             session)

assert counter == master_inpt.shape[0]

data_dir = 'data_for_cluster/'



json = json.dumps(animal_eid_dict)
f = open("partially_processed/animal_eid_dict.json", "w")
f.write(json)
f.close()

np.savez('partially_processed/animal_list.npz', animal_list)
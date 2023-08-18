# Download IBL dataset and begin processing it: identify unique animals in
# IBL dataset that enter biased blocks.  Save a dictionary with each animal
# and a list of their eids in the biased blocks

import numpy as np
from one.api import ONE, One
import numpy.random as npr
import json
from collections import defaultdict
import wget
import pickle
from pprint import pprint
from zipfile import ZipFile
import os
npr.seed(65)

DOWNLOAD_DATA = False  # change to True to download raw data (WARNING: this
# can take a while)

if __name__ == '__main__':
    ibl_data_path = "../../data/ibl"
    cache_dir = ibl_data_path + "/ibl-behavioral-data-Dec2019"
    if DOWNLOAD_DATA: # Warning: this step takes a while
        if not os.path.exists(ibl_data_path):
            os.makedirs(ibl_data_path)
        # download IBL data
        url = 'https://ndownloader.figshare.com/files/21623715'
        wget.download(url, ibl_data_path)
        # now unzip downloaded data:
        with ZipFile(ibl_data_path + "/ibl-behavior-data-Dec2019.zip",
                     'r') as zipObj:
            # extract all the contents of zip file in ibl_data_path
            zipObj.extractall(ibl_data_path)

        # after downloading the data, create a cache
        print('Building ONE cache from filesystem...')
        One.setup(cache_dir, hash_files=False)

    # set ONE in local mode using data in the cache directory
    one = ONE(cache_dir=cache_dir)
    print(one)

    part_proc_dir = ibl_data_path + "/partially_processed"
    # create directory for saving data:
    if not os.path.exists(part_proc_dir):
        os.makedirs(part_proc_dir)

    # search with ONE
    # eids = one.search(['_ibl_trials.*'])
    eids, info = one.search(dataset=['trials'], details=True)
    assert len(eids) > 0, "ONE search is in incorrect directory"
    animal_list = []
    animal_eid_dict = defaultdict(list)

    for i, (eid, session_info) in enumerate(zip(eids, info)):
        print("\n================================================")
        print(f"{i +1} of {len(eids)}")
        print("------------------------------------------------")
        print(eid)
        print("------------------------")
        pprint(session_info)

        bias_probs = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
        comparison = np.unique(bias_probs) == np.array([0.2, 0.5, 0.8])
        # sessions with bias blocks
        if isinstance(comparison, np.ndarray):
            # update def of comparison to single True/False
            comparison = comparison.all()
        if comparison == True:
            animal = session_info['subject']
            if animal not in animal_list:
                animal_list.append(animal)
            animal_eid_dict[animal].append(eid)

    json = json.dumps(animal_eid_dict)
    f = open(part_proc_dir + "/animal_eid_dict.json",  "w")
    f.write(json)
    f.close()

    np.savez(part_proc_dir + "/animal_list.npz", animal_list)

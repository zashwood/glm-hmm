# Download IBL dataset and begin processing it: identify unique animals in
# IBL dataset that enter biased blocks.  Save a dictionary with each animal
# and a list of their eids in the biased blocks

import numpy as np
from oneibl.onelight import ONE
import numpy.random as npr
import json
from collections import defaultdict
import wget
from zipfile import ZipFile
import os
from preprocessing_utils import get_animal_name
npr.seed(65)

DOWNLOAD_DATA = True  # change to True to download raw data (WARNING: this
# can take a while)

if __name__ == '__main__':
    ibl_data_path = "../../data/ibl/"
    if DOWNLOAD_DATA: # Warning: this step takes a while
        if not os.path.exists(ibl_data_path):
            os.makedirs(ibl_data_path)
        # download IBL data
        url = 'https://ndownloader.figshare.com/files/21623715'
        wget.download(url, ibl_data_path)
        # now unzip downloaded data:
        with ZipFile(ibl_data_path + "ibl-behavior-data-Dec2019.zip",
                     'r') as zipObj:
            # extract all the contents of zip file in ibl_data_path
            zipObj.extractall(ibl_data_path)

    # create directory for saving data:
    if not os.path.exists(ibl_data_path + "partially_processed/"):
        os.makedirs(ibl_data_path + "partially_processed/")

    # change directory so that ONE searches in correct directory:
    os.chdir(ibl_data_path)
    one = ONE()
    eids = one.search(['_ibl_trials.*'])
    assert len(eids) > 0, "ONE search is in incorrect directory"
    animal_list = []
    animal_eid_dict = defaultdict(list)

    for eid in eids:
        bias_probs = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
        comparison = np.unique(bias_probs) == np.array([0.2, 0.5, 0.8])
        # sessions with bias blocks
        if isinstance(comparison, np.ndarray):
            # update def of comparison to single True/False
            comparison = comparison.all()
        if comparison == True:
            animal = get_animal_name(eid)
            if animal not in animal_list:
                animal_list.append(animal)
            animal_eid_dict[animal].append(eid)

    json = json.dumps(animal_eid_dict)
    f = open("partially_processed/animal_eid_dict.json",  "w")
    f.write(json)
    f.close()

    np.savez('partially_processed/animal_list.npz', animal_list)

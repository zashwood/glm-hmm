# Obtain IBL response time data for producing Figure 6
# Write out the response times and the corresponding sessions
import os

import numpy as np
import numpy.random as npr
from oneibl.onelight import ONE

from preprocessing_utils import load_animal_eid_dict, load_data

npr.seed(65)

if __name__ == '__main__':
    ibl_data_path = "../../data/ibl/"
    animal_eid_dict = load_animal_eid_dict(
        ibl_data_path + 'data_for_cluster/final_animal_eid_dict.json')
    # must change directory for working with ONE
    os.chdir(ibl_data_path)
    one = ONE()

    data_dir = 'response_times/data_by_animal/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for animal in animal_eid_dict.keys():
        print(animal)
        animal_inpt, animal_y, animal_session = load_data(
            'data_for_cluster/data_by_animal/' + animal + '_processed.npz')
        for z, eid in enumerate(animal_eid_dict[animal]):
            raw_session_id = eid.split('Subjects/')[1]
            session_id = raw_session_id.replace('/', '-')
            full_sess_len = len(one.load_dataset(eid, '_ibl_trials.choice'))

            file_names = [
                '_ibl_trials.feedback_times', '_ibl_trials.response_times',
                '_ibl_trials.goCue_times', '_ibl_trials.stimOn_times'
            ]

            save_vars = [
                'feedback_times', 'response_times', 'go_cues', 'stim_on_times'
            ]

            for i, file in enumerate(file_names):
                full_path = 'ibl-behavioral-data-Dec2019/' + eid + \
                                 '/alf/' + file + '.npy'
                if os.path.exists(full_path):
                    globals()[save_vars[i]] = one.load_dataset(eid, file)
                else:
                    globals()[save_vars[i]] = np.empty((full_sess_len, ))
                    globals()[save_vars[i]][:] = np.nan

            start = np.nanmin(np.c_[stim_on_times, go_cues], axis=1)

            if (len(feedback_times) == len(response_times)): # some response
                # times/feedback times are missing, so fill these as best as
                # possible
                end = np.nanmin(np.c_[feedback_times, response_times], axis=1)
            elif len(feedback_times) == full_sess_len:
                end = feedback_times
            elif len(response_times) == full_sess_len:
                end = response_times

            # check timestamps increasing:
            idx_to_change = np.where(start > end)[0]

            if len(idx_to_change) > 0:
                start[idx_to_change[0]] = np.nan
                end[idx_to_change[0]] = np.nan

            # Check we have times for at least some trials
            nan_trial = np.isnan(np.c_[start, end]).any(axis=1)

            is_increasing = (((start < end) | nan_trial).all() and
                    ((np.diff(start) > 0) | np.isnan(
                        np.diff(start))).all())

            if is_increasing and ~nan_trial.all() and len(start) == \
                    full_sess_len and len(end) == full_sess_len: #
                # check that times are increasing and that len(start) ==
                # full_sess_len etc
                prob_left_dta = one.load_dataset(
                    eid, '_ibl_trials.probabilityLeft')
                assert start.shape[0] == prob_left_dta.shape[0],\
                    "different lengths for prob left and raw response dta: " + \
                    str(start.shape[0]) + " vs " + str(
                        prob_left_dta.shape[0])

                # subset to trials corresponding to prob_left == 0.5:
                unbiased_idx = np.where(prob_left_dta == 0.5)
                response_dta = end[unbiased_idx] - start[unbiased_idx]

                if ((np.nanmedian(response_dta) >= 10) | (np.nanmedian(
                        response_dta) == np.nan)): # check that median
                    # response time for session is less than 10 seconds
                    response_dta = np.array([np.nan for i in range(len(
                        unbiased_idx[0]))])

                rt_sess = [session_id for i in range(response_dta.shape[0])]
                # before saving, confirm that there are as many trials as in
                # some of the other data:
                assert len(rt_sess) == animal_inpt[np.where(animal_session ==
                                                            session_id),
                                       :].shape[1], "response dta is different " \
                                                    "shape compared to inpt"
            else: # if any of the conditions above fail, fill the session's
                # data with nans
                len_prob_50 = animal_inpt[np.where(animal_session ==
                                                            session_id),
                              :].shape[1]
                response_dta = np.array([np.nan for i in range(len_prob_50)])
                rt_sess = [session_id for i in range(response_dta.shape[0])]

            if z == 0:
                rt_session_dta_this_animal = rt_sess
                response_dta_this_animal = response_dta
            else:
                rt_session_dta_this_animal = np.concatenate(
                    (rt_session_dta_this_animal, rt_sess))
                response_dta_this_animal = np.concatenate(
                    (response_dta_this_animal, response_dta))

        assert len(response_dta_this_animal) == len(animal_inpt), "different size for response times and inpt"
        np.savez(data_dir + animal + '.npz', response_dta_this_animal,
                 rt_session_dta_this_animal)

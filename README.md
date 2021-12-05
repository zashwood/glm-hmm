# glm-hmm
Code to reproduce figures in ["Mice alternate between discrete strategies during perceptual decision-making"](https://www.biorxiv.org/content/10.1101/2020.10.19.346353v3.full.pdf) from Ashwood, Roy, Stone, IBL, Urai, Churchland, Pouget and Pillow (2020).

Code is ordered so that data is preprocessed into desired format according the scripts in "1_preprocess_data". 
Next, you can run the models discussed in the paper (both the GLM-HMM, as well as the classic lapse model) using the code contained in "2_fit_models".
Finally, assuming that you have downloaded and preprocessed the datasets, and that you have fit the models on these datasets, 
you can reproduce the figures shown in our paper by running the code contained in "3_make_figures".
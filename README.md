# CS75: Introduction to Bioinformatics
Seok Jun Bing, Tom Hao Chang, Jing Geng, Deven Orie, and Juan Camilo Torres

## Set Up
*Developed in Python 2.7 environment<br>*
Git pull and download:
* [all_plants.fas_updated04152015](http://bioinformatics.ysu.edu/publication/data/PlantSecKB/)
file to ./data/plants/
* [metazoa_proteins.fas](http://proteomics.ysu.edu/publication/data/MetazSecKB/) file to ./data/animals/

## Data Files
* `./data/aaindex/aaindex1.txt` contains raw dictionary data used by functions in `read_dicts.py`. **Don't alter**.
* `./data/aaindex/aaindex_used.txt` is a list of aaindices to use when calculating values of features. Feel to alter this
file.
* `./data/aaindex/aaindex_all.txt` is a list of all 566 aaindices. Used just as a backup in case you want to repopulate 
`aaindex_used.txt`. **Don't alter**.
* `./data/aaindex/list_of_indices.txt` Index codes and their names for easy parsing. **Don't alter**.
* `./data/aaindex/aaindex_used_animal_top10.txt` Top 10 index selection for Animal data
* `./data/aaindex/aaindex_used_animal_top20.txt` Top 20 index selection for Animal data
* `./data/aaindex/aaindex_used_plant_top10.txt` Top 10 index selection for Plant data
* `./data/aaindex/aaindex_used_plant_top20.txt` Top 20 index selection for Plant data
* `./data/aaindex/aindex_used_low_corr.txt` Top 123 <0.8 correlation index selection

* `./data/animals/permutation_testing.csv` output from permutation testing
* `./data/compare/` This directory contains data from a comparison study

## Data Preprocessing
* `./models/` contains weights of a trained LSTM
* `./src/data_processing/read_data.py` contains functions to process raw data inputs such as all_plants.fas_updated04152015
and outputs a processed data file `./data/plants/label_scores.txt` which contains the labels and the score values in the 
order of the aaindices in `aaindex_used.txt`
    * run by `python read_data.py [plants/animals] [scores/sequences/kscores]`
        * `[plant/animals]` is an option to choose between plants and animals data
        * `[scores/sequences/kscores]` is an option to choose to one of the following outputs
            * `scores` outputs subcellular locations and feature scores
            * `sequences` outputs subcellular locations and amino acid sequences
            * `kscores` outputs perfectly balanced set of subcellular locations and feature scores        
* `./src/data_processing/read_dicts.py` contains functions to build dictionaries. These functions are called in `read_data.py`.

## Predictive Algorithms
* `./src/ml/feature_importance_impurity` generates a graph ranking indices by Gini impurity
    * run by invoking `python feature_importance_impurity.py`
* `./src/ml/keras_nn.py` is a densely connected neural network
    * run by invoking `python keras_nn.py`
* `./src/ml/lstm.py` lstm written in keras
* `./src/ml/permutation_testing.py` permutation testing of labels
    * run by invoking `python permutation_testing.py [run/read]` where the run option reads and writes data to csv and read option reads the csv and plots 
* `./src/ml/random_forest.py` script to test performance of random forest
* `./src/ml/svm.py` script to test performance of linear SVM
* `./src/ml/test_graphs.py` script to print out the ROC graph
* `./src/ml/voting_classifier.py` runs Voting Classifier
    * run by invoking `python voting_classifier.py [compare/default]`
        * `compare` options runs voting classifier on data from a comparison study
        * `default` options runs voting classifier on our own data
            * **before running this script please make sure to run `read_data.py` first to generate the relevant input data files**

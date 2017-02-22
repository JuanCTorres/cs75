# CS75: Introduction to Bioinformatics
Seok Jun Bing, Tom Hao Chang, Jing Geng, Deven Orie, and Juan Camilo Torres

## Set Up
Git pull and download:
* [all_plants.fas_updated04152015](http://bioinformatics.ysu.edu/publication/data/PlantSecKB/)
file to ./data/plants/
* [metazoa_proteins.fas](http://proteomics.ysu.edu/publication/data/MetazSecKB/) file to ./data/animals/ (not yet used...)

## File description
* ./data/aaindex/aaindex1.txt contains raw dictionary data used by functions in read_dicts.py. Don't alter.
* ./data/aaindex/aaindex_used.txt is a list of aaindices to use when calculating values of features. Feel to alter this
file.
* ./data/aaindex/aaindex_all.txt is a list of all 566 aaindices. Used just as a backup in case you want to repopulate 
aaindex_used.txt. Don't alter.
* ./src/data_processing/read_data.py contains functions to process raw data inputs such as all_plants.fas_updated04152015
and outputs a processed data file ./data/plants/label_scores.txt which contains the labels and the score values in the 
order of the aaindices in aaindex_used.txt
* ./src/data_processing/read_dicts.py contains functions to build dictionaries. These functions are called in read_data.py.



# How to process data
* run read_data.py
* 'size' variable in \_\_main__ determines the number of data points in output file ./data/plants/label_scores.txt.
Feel free to change this variable.

# Reading in preprocessed data
use read_data.read_preprocessed_data() which returns a tuple of labels and features

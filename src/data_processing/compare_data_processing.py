from Bio import SeqIO
from os import listdir
import pandas as pd
from data_processing.read_dicts import construct_dicts

"""
This file should be used for reading the data files from other papers we are comparing the results to.
This is required since their formats are somewhat different
"""

# FOLDERS
DATA_FOLDER = '../../data'
AAINDEX_FOLDER = '%s/aaindex' % DATA_FOLDER
COMPARE_FOLDER = '%s/compare' % DATA_FOLDER

# FILES
AAINDEX_FILE = '%s/aaindex1.txt' % AAINDEX_FOLDER
AAINDEX_USED_FILE = '%s/aaindex_used.txt' % AAINDEX_FOLDER


def read_all_fasta_files(dir):
    """
    Reads all the fasta files in a directory and puts them in a dataframe.

    :param dir: directory where all the files are
    :return: dataframe w/ each point labeled by its filename
    """
    files_and_labels = [('%s/%s' % (COMPARE_FOLDER, f), f.split('.')[0]) for f in listdir(COMPARE_FOLDER)]
    sequence_matrix = [(SeqIO.parse(file, 'fasta'), label) for file, label in files_and_labels]
    cols = ['seq', 'label']
    main_df = pd.DataFrame(columns=cols)

    for sequence_list, label in sequence_matrix:
        sequence_list = list(sequence_list)
        label_list = [label for i in range(len(sequence_list))]
        current_label_df = pd.DataFrame(data=[[str(line.seq) for line in sequence_list], label_list])
        current_label_df = current_label_df.transpose()
        current_label_df.columns = cols
        main_df = main_df.append(current_label_df)

    return main_df


def get_scores(sequence_matrix, scoring_file, indices_used):
    """
    Takes a DataFrame of sequences and adds one score for each index used to each sequence.

    :param sequence_matrix: DataFrame containing sequences and labels
    :param scoring_file: string of file path of the aaindex scoring file
    :param indices_used: list of index accession names for the indices to be used to score the sequences
    :return: sequence_matrix with added rows, one for each index in indices_used
    """
    index_scores = construct_dicts(scoring_file)[0]
    for index in indices_used:
        scores = []
        for seq in sequence_matrix.seq:
            val = sum([index_scores.get(index).get(char, 0) for char in seq])
            scores.append(val)
        sequence_matrix[index] = pd.Series(scores)
    return sequence_matrix


if __name__ == '__main__':
    with open(AAINDEX_USED_FILE) as f:
        indices_used = [line.strip() for line in f.readlines()]
    print(indices_used)
    mat = read_all_fasta_files(COMPARE_FOLDER)
    scores = get_scores(mat, AAINDEX_FILE, indices_used)
    print(scores.head())

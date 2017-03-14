from read_data import read_preprocessed_data, write_label_score_file
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import sys


def calculate_mean_sd(data_file, features_file):
    # print 'adsada'
    with open(features_file, 'r') as f:
        indices = [line.strip() for line in f.readlines()]

    labels, data = read_preprocessed_data(data_file, features_file)
    assert len(labels) == len(data)

    # data = StandardScaler().fit_transform(data)

    d = dict()

    for ind, val in enumerate(data):
        try:
            d[labels[ind]].append(val)
        except KeyError:
            d[labels[ind]] = [val]

    for k in d.keys():
        d[k] = np.asarray(d[k])

    d_mean = dict()
    d_sd = dict()
    for ind, val in enumerate(indices):
        d_mean[val] = list()
        d_sd[val] = list()
        for k in d.keys():  # location
            d_mean[val].append(d[k].mean(axis=0)[ind])
            d_sd[val].append(d[k].std(axis=0)[ind])

    plt.figure(1)
    plt.suptitle("Mean Scores and Distribution of aaIndex", fontsize=16)

    # plt.xticks(range(len(d.keys())), d.keys(), rotation='vertical')
    plt.subplot(311)
    plt.errorbar(range(len(d.keys())), d_mean['PUNT030101'], d_sd['PUNT030101'], linestyle='None', marker='*', label='PUNT030101')
    # plt.xticks(range(len(d.keys())), ['' for i in range(len(d.keys()))], rotation='vertical')
    plt.tick_params(axis='x', labelbottom='off')
    plt.title('PUNT030101')
    plt.ylabel('Score')

    plt.subplot(312)
    plt.errorbar(range(len(d.keys())), d_mean['KLEP840101'], d_sd['KLEP840101'], linestyle='None', marker='*',label='KLEP840101')
    # plt.xticks(range(len(d.keys())), ['' for i in range(len(d.keys()))], rotation='vertical')
    plt.tick_params(axis='x', labelbottom='off')
    plt.title('KLEP840101')
    plt.ylabel('Score')

    plt.subplot(313)
    plt.errorbar(range(len(d.keys())), d_mean['OOBM850102'], d_sd['OOBM850102'], linestyle='None', marker='*', label='OOBM850102')
    plt.xticks(range(len(d.keys())), d.keys(), rotation='vertical')
    plt.title('OOBM850102')
    plt.ylabel('Score')

    plt.show()

    plt.figure(2)
    # plotting x = indices, y = scores, std
    plt.xticks(range(len(indices)), indices, rotation='vertical')
    for k in d.keys():
        d[k] = np.asarray(d[k])
        print 'mean: %s: %s' % (k, str(d[k].mean(axis=0)))
        print 'std: %s: %s' % (k, str(d[k].std(axis=0)))

        plt.errorbar(range(len(indices)), d[k].mean(axis=0), d[k].std(axis=0), linestyle='None', marker='^', label=k)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    ENABLE_WRITE = 1
    SIZE = 100000
    FEATURES_FILE = '../../data/aaindex/aaindex_used.txt'

    if len(sys.argv) == 2:
        dataset = sys.argv[1]
    else:
        sys.exit(1)

    if dataset == 'plants':
        data_folder = '../../data/plants'
        input_file = '%s/all_plants.fas_updated04152015' % data_folder
    elif dataset == 'animals':
        data_folder = '../../data/animals'
        input_file = '%s/metazoa_proteins.fas' % data_folder
    else:
        raise Exception('Please enter a valid dataset to use. Accepted: \'plants\' and \'animals\'')

    output_file_1 = '%s/label_scores.txt' % data_folder

    # if os.path.exists(output_file_1) and ENABLE_WRITE != 0:
    #     os.remove(output_file_1)
    # write_label_score_file(input_file, output_file_1, write_file=ENABLE_WRITE, outsize=SIZE, group_similar_labels=True)

    calculate_mean_sd(output_file_1, FEATURES_FILE)


    pass
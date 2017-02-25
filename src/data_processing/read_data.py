import os
import re
from collections import Counter
import read_dicts
import pandas as pd


def getscores(d, aalist, seq):
    score_list = list()

    char_freq = dict()

    for c in seq:
        if c in char_freq:
            char_freq[c] += 1
        else:
            char_freq[c] = 1

    for aa in aalist:
        score = 0

        for k in d[aa].iterkeys():
            try:
                freq = char_freq[k]
            except KeyError:
                freq = 0

            score += d[aa][k] * freq

        score_list.append(str(score))

    return '|'.join(score_list)


def get_specific_label(line):
    location_search = re.search(r"(.+(\[)(?P<location1>.+?)(\])( |)$)", line)
    location = location_search.group('location1').rstrip().split(',')[0]
    return location


def get_general_label(line):
    location_search = re.search(r"(.+(\[)(?P<location1>.+?)(\])( |)$)", line)
    try:
        location = location_search.group('location1')
        # funny looking because animal and plants formatting differs
        general_location = location.split('(')[0].rstrip().split(',')[0]
    except AttributeError:
        print('line: ' + line)
        # print('location: ' + location)
        assert False
    return general_location


def get_general_label_test(file_in):
    d = dict()
    with open(file_in, 'r') as ifile:
        for line in ifile:
            if line[0] == '>':
                # print(line)
                location = get_general_label(line)

                if location in d.keys():
                    d[location] += 1
                else:
                    d[location] = 1

    for k, v in d.items():
        print('k: %-30s v: %d' % (k, v))


def write_label_score_file(file_in, file_out, write_file=0, outsize='all', group_label=True):
    print('building and writing %s' % file_out)

    count = 0
    entry_count = 0

    score_d, corr_d = read_dicts.construct_dicts("../../data/aaindex/aaindex1.txt")
    aalist = read_dicts.get_aaindex_list("../../data/aaindex/aaindex_used.txt")

    with open(file_in, 'r') as ifile:
        for i, l in enumerate(ifile):
            count = i + 1
        print('raw data lines: %d' % count)
    with open(file_in, 'r') as ifile:
        with open(file_out, 'a') as ofile:
            for i in range(count):
                # print "%d of %d lines" % (i+1, count)
                l = ifile.readline()
                # if i == 1000:
                #     break
                if l[0] == '>':
                    if group_label:
                        location = get_general_label(l)
                    else:
                        location = get_specific_label(l)

                    # location_search = re.search(r".+(\[)(?P<location>.+?)(\])$", l)
                    # location = location_search.group('location').rstrip()
                    # print location

                else:
                    seq = ''
                    seq += l.rstrip()
                    while True:
                        x = ifile.tell()
                        l = ifile.readline()

                        if l == '':  # EOF
                            # do something
                            # print seq
                            if (location != 'NULL') and (location != '\N') and (write_file != 0):
                                scores = getscores(score_d, aalist, seq)
                                ofile.write('%s|%s\n' % (location, scores))
                                entry_count += 1
                                print('number of entries: %d' % entry_count)
                            del seq

                            return
                        elif l[0] == '>':
                            ifile.seek(x)
                            break
                        else:
                            seq += l.rstrip()
                    # do something
                    # print seq + '\n'

                    if (location != 'NULL') and ('\N' not in location) and (write_file != 0):
                        scores = getscores(score_d, aalist, seq)
                        ofile.write('%s|%s\n' % (location, scores))
                        entry_count += 1
                        print('number of entries: %d' % entry_count)
                        if outsize != 'all':
                            if entry_count == outsize:
                                break
                    del seq


def write_label_seq_file(file_in, file_out, write_file=0):
    count = 0
    with open(file_in, 'r') as ifile:
        for i, l in enumerate(ifile):
            count = i + 1
        print('num lines: %d' % count)
    with open(file_in, 'r') as ifile:
        with open(file_out, 'a') as ofile:
            for i in range(count):
                l = ifile.readline()

                # if i == 1000:
                #     break
                if l[0] == '>':
                    # id_search = re.search(r"\|(?P<id>.+?)\|", l)
                    # id = id_search.group('id').rstrip()
                    # print id
                    #
                    # name_search = re.search(r"(\| )(?P<name>.+?)( \()", l)
                    # name = name_search.group('name').rstrip()
                    # print name

                    location_search = re.search(r".+(\[)(?P<location>.+?)(\])$", l)
                    location = location_search.group('location').rstrip()
                    print(location)

                else:
                    seq = ''
                    seq += l.rstrip()
                    while True:
                        x = ifile.tell()
                        l = ifile.readline()

                        if l == '':  # EOF
                            # do something
                            # print seq
                            if location != 'NULL' and write_file != 0:
                                ofile.write('%s|%s\n' % (location, seq))
                            del seq

                            return
                        elif l[0] == '>':
                            ifile.seek(x)
                            break
                        else:
                            seq += l.rstrip()
                    # do something
                    # print seq + '\n'
                    if location != 'NULL' and write_file != 0:
                        ofile.write('%s|%s\n' % (location, seq))
                    del seq


def find_unique_labels(filename):
    with open(filename, 'r') as ifile:

        d = dict()
        for l in ifile:
            label = l.strip().split('|')[0]

            if label in d:
                d[label] += 1
            else:
                d[label] = 1

        for k, v in d.iteritems():
            print('k: %-30s v: %d' % (k, v))


def check_label_seq_file_validity(filename):
    print("\nchecking validity of output file...")
    non_alpha_count = 0
    invalid_label_count = 0
    invalid_chars = ['[', ']', '-', ',', '.', '|', '\\']
    with open(filename, 'r') as ifile:

        for l in ifile:
            label, seq = l.strip().split('|')

            if not seq.isalpha():
                print("non alpha detected in seq!")
                non_alpha_count += 1
                for i in seq:
                    if not i.isalpha():
                        print(i)

            if any(c in label for c in invalid_chars):
                invalid_label_count += 1
                print(label)

        if non_alpha_count != 0 or invalid_label_count != 0:
            raise Exception("output file not valid")
        else:
            print("\noutput file seems fine\n")


def read_preprocessed_data(input_file, features_file, exclude_labels_less_than=0, format='default'):
    """
    reads in label_scores.txt file and returns the labels and features as lists or dataframes
    :param input_file: directory of label_scores.txt file
    :param exclude_labels_less_than: skip labels with occurrence less than this value
    :param format default or df: what format to return the labels and features in
    :return: (labels, features)
    """
    with open(input_file, 'r') as ifile:
        lines = [line.rstrip().split('|') for line in ifile.readlines()]
    with open(features_file, 'r') as f:
        features_used = [line.strip() for line in f.readlines()]

    all_labels = [line[0] for line in lines]
    occurrences = Counter(all_labels)
    labeled_data = [(lines[i][0], map(float, lines[i][1:])) for i in range(len(lines)) if
                    occurrences[all_labels[i]] >= exclude_labels_less_than]
    labels, feature_matrix = zip(*labeled_data)

    if format == 'default':
        return list(labels), list(feature_matrix)  # tuples can make some things harder
    elif format == 'df':
        data = pd.DataFrame(data=list(feature_matrix), columns=features_used)
        labels = pd.DataFrame(data=list(labels))
        return labels, data
    else:
        raise Exception('Unknown format %s' % format)


if __name__ == '__main__':
    # PLANTS
    # INPUT_FILE = "../../data/plants/all_plants.fas_updated04152015"
    # OUTPUT_FILE0 = "../../data/plants/label_seq.txt"
    # OUTPUT_FILE1 = "../../data/plants/label_scores.txt"

    # ANIMALS
    INPUT_FILE = "../../data/animals/metazoa_proteins.fas"
    OUTPUT_FILE0 = "../../data/animals/label_seq.txt"
    OUTPUT_FILE1 = "../../data/animals/label_scores.txt"

    ENABLE_WRITE = 1

    # number of entries to output in the label & scores file.... max is 1257123
    size = 100000

    # testing
    # get_general_label_test(input_file)

    # UNCOMMENT THIS BLOCK TO OUTPUT LABEL & SEQUENCE file
    # if os.path.exists(output_file0) and enable_write != 0:
    #     os.remove(output_file0)
    # write_label_seq_file(input_file, output_file0, write_file=enable_write)
    # find_unique_labels(output_file0)
    # check_label_seq_file_validity(output_file0)

    # UNCOMMENT THIS BLOCK TO OUTPUT LABEL & SCORES file
    if os.path.exists(OUTPUT_FILE1) and ENABLE_WRITE != 0:
        os.remove(OUTPUT_FILE1)
    write_label_score_file(INPUT_FILE, OUTPUT_FILE1, write_file=ENABLE_WRITE, outsize=size, group_label=True)
    print('\n%s contains these labels:' % OUTPUT_FILE1)
    find_unique_labels(OUTPUT_FILE1)

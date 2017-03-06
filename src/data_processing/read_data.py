import os, re, sys
import read_dicts
from collections import Counter
import pandas as pd
import operator

sys.path.append('.')

ENABLE_WRITE = 1
INDEX_NAMES_FILES = '../../data/aaindex/list_of_indices.txt'


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


def get_species(line):
    location_search = re.search(r"\(sp: (?P<location1>.+?)\)", line)
    location = location_search.group('location1').rstrip()
    return location


def write_label_score_file(file_in, file_out, write_file=0, outsize='all', group_similar_labels=True,
                           species='all'):
    print('building and writing %s' % file_out)

    count = 0
    entry_count = 0
    duplicate_count = 0
    uniques = set()
    d_sp = dict()

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
                    if group_similar_labels:
                        location = get_general_label(l)
                    else:
                        location = get_specific_label(l)

                    sp = get_species(l)

                else:
                    seq = ''
                    seq += l.rstrip()
                    while True:
                        x = ifile.tell()
                        l = ifile.readline()

                        if l == '':  # EOF
                            # do something
                            # print seq
                            if (location != 'NULL') and (location != '\N') and (seq not in uniques) and (write_file != 0):
                                if species == 'all' or species == sp:
                                    try:
                                        d_sp[sp] += 1
                                    except KeyError:
                                        d_sp[sp] = 1

                                    uniques.add(seq)
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
                    # if seq in uniques:
                    #     duplicate_count += 1
                    #     print 'found dup:' + location + ' ' + seq
                    #     print duplicate_count

                    if (location != 'NULL') and ('\N' not in location) and (seq not in uniques) and (write_file != 0):
                        if species == 'all' or species == sp:

                            try:
                                d_sp[sp] += 1
                            except KeyError:
                                d_sp[sp] = 1

                            uniques.add(seq)
                            scores = getscores(score_d, aalist, seq)
                            ofile.write('%s|%s\n' % (location, scores))
                            entry_count += 1
                            print('number of entries: %d' % entry_count)
                            if outsize != 'all':
                                if entry_count == outsize:
                                    print 'dic:'
                                    sorted_x = sorted(d_sp.items(), key=operator.itemgetter(1))
                                    print sorted_x
                                    break

                        # else:
                        #     print 'anh'
                    del seq


def write_sequence_file(file_in, file_out, write_file=0, outsize='all', group_similar_labels=True, species='all'):
    print('building and writing %s' % file_out)
    location_set, max_len, seq_count, long_count = set(), 0, 0, 0
    count = 0
    entry_count = 0
    uniques = set()
    duplicate_count = 0
    d_sp = dict()

    with open(file_in, 'r') as ifile:
        for i, l in enumerate(ifile):
            count = i + 1
        print('raw data lines: %d' % count)
    with open(file_in, 'r') as ifile:
        with open(file_out, 'a') as ofile:
            for i in range(count):
                l = ifile.readline()
                if l[0] == '>':
                    if group_similar_labels:
                        location = get_general_label(l)
                    else:
                        location = get_specific_label(l)

                    sp = get_species(l)

                else:
                    seq = ''
                    seq += l.rstrip()
                    while True:
                        x = ifile.tell()
                        l = ifile.readline()

                        if l == '':  # EOF
                            if (location != 'NULL') and (location != '\N') and (seq not in uniques) and (write_file != 0):
                                if species == 'all' or species == sp:
                                    try:
                                        d_sp[sp] += 1
                                    except KeyError:
                                        d_sp[sp] = 1

                                    uniques.add(seq)
                                    ofile.write('%s|%s\n' % (location, seq))
                                    if len(seq) <= 500:
                                        long_count += 1
                                    location_set.add(location)
                                    entry_count += 1
                                    print('number of entries: %d' % entry_count)
                            del seq

                            return
                        elif l[0] == '>':
                            ifile.seek(x)
                            break
                        else:
                            seq += l.rstrip()

                    # if seq in uniques:
                    #     duplicate_count += 1
                    #     print 'found dup:' + location + ' ' + seq
                    #     print duplicate_count

                    if (location != 'NULL') and ('\N' not in location) and (seq not in uniques) and (write_file != 0):
                        if species == 'all' or species == sp:
                            try:
                                d_sp[sp] += 1
                            except KeyError:
                                d_sp[sp] = 1

                            uniques.add(seq)
                            ofile.write('%s|%s\n' % (location, seq))
                            if len(seq) <= 500:
                                long_count += 1
                            location_set.add(location)
                            entry_count += 1
                            print('number of entries: %d' % entry_count)
                            if outsize != 'all':
                                if entry_count == outsize:
                                    print 'dic:'
                                    sorted_x = sorted(d_sp.items(), key=operator.itemgetter(1))
                                    print sorted_x
                                    break
                    del seq
    print("locations: " + str(location_set))
    print("maximum sequence length: " + str(max_len))
    print("Total sequences: " + str(entry_count))
    print("Long sequences: " + str(long_count))


# def write_label_seq_file(file_in, file_out, write_file=0):
#     count = 0
#     with open(file_in, 'r') as ifile:
#         for i, l in enumerate(ifile):
#             count = i + 1
#         print('num lines: %d' % count)
#     with open(file_in, 'r') as ifile:
#         with open(file_out, 'a') as ofile:
#             for i in range(count):
#                 l = ifile.readline()
#
#                 # if i == 1000:
#                 #     break
#                 if l[0] == '>':
#
#                     location_search = re.search(r".+(\[)(?P<location>.+?)(\])$", l)
#                     location = location_search.group('location').rstrip()
#                     print(location)
#
#                 else:
#                     seq = ''
#                     seq += l.rstrip()
#                     while True:
#                         x = ifile.tell()
#                         l = ifile.readline()
#
#                         if l == '':  # EOF
#                             # do something
#                             # print seq
#                             if location != 'NULL' and write_file != 0:
#                                 ofile.write('%s|%s\n' % (location, seq))
#                             del seq
#
#                             return
#                         elif l[0] == '>':
#                             ifile.seek(x)
#                             break
#                         else:
#                             seq += l.rstrip()
#                     # do something
#                     # print seq + '\n'
#                     if location != 'NULL' and write_file != 0:
#                         ofile.write('%s|%s\n' % (location, seq))
#                     del seq


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


def get_index_names(index_code_list, index_info_file=INDEX_NAMES_FILES):
    """

    Returns the names and descriptions for a list of index codes

    :param index_code_list: List of index codes, e.g. ['RADA880101','WILM950102']
    :param index_info_file: File containing the names and descriptions of the indices.
    :return: names and descriptions of the index codes in `index_code_list`
    >>> ind = get_index_names(['RADA880101', 'BIOV880101', 'SNEP660102','WILM950102']); get_index_names(ind)
    """
    with open(index_info_file, 'r') as f:
        lines = [line.strip().split(' ', 1) for line in f.readlines()]
        lines = [line for line in lines if len(line) > 0]
    code_dict = {code: desc for code, desc in lines}
    return {name: code_dict[name] for name in index_code_list}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        mode = sys.argv[2]
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

    if mode == 'scores':
        pass
    elif mode == 'sequences':
        pass
    else:
        raise Exception('Please enter either "scores" or "sequences" as the second argument')

    output_file_0 = '%s/label_seq.txt' % data_folder
    output_file_1 = '%s/label_scores.txt' % data_folder
    output_file_2 = '%s/label_sequences.txt' % data_folder

    # number of entries to output in the label & scores file.... max is 1257123


    # testing
    # get_general_label_test(input_file)

    # UNCOMMENT THIS BLOCK TO OUTPUT LABEL & SEQUENCE file
    # if os.path.exists(output_file_0) and ENABLE_WRITE != 0:
    #     os.remove(output_file_0)
    # write_label_seq_file(input_file, output_file_0, write_file=0)
    # # find_unique_labels(output_file_0)
    # check_label_seq_file_validity(output_file_0)

    # UNCOMMENT THIS BLOCK TO OUTPUT LABEL & SCORES file
    if mode == 'scores':
        if os.path.exists(output_file_1) and ENABLE_WRITE != 0:
            os.remove(output_file_1)
        size = 500000
        write_label_score_file(input_file, output_file_1, write_file=ENABLE_WRITE, outsize=size,
                               group_similar_labels=True, species='all')   # species = 'all' for everything
        # 'Rattus norvegicus', 7071), ('Mus musculus', 15461), ('Homo sapiens', 23931) are popular species
        print('\n%s contains these labels:' % output_file_1)
        find_unique_labels(output_file_1)
    if mode == 'sequences':
        if os.path.exists(output_file_2) and ENABLE_WRITE != 0:
            os.remove(output_file_2)
        size = 100000
        write_sequence_file(input_file, output_file_2, write_file=ENABLE_WRITE, outsize=size, group_similar_labels=True,
                            species='all')     # species = 'all' for everything
        # 'Rattus norvegicus', 7071), ('Mus musculus', 15461), ('Homo sapiens', 23931) are popular species

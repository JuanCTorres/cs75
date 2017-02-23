import os
import re
from collections import Counter
import read_dicts


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


def write_label_score_file(file_in, file_out, write_file=0, outsize='all'):
    print 'building and writing %s' % file_out

    count = 0
    entry_count = 0

    score_d, corr_d = read_dicts.construct_dicts("../../data/aaindex/aaindex1.txt")
    aalist = read_dicts.get_aaindex_list("../../data/aaindex/aaindex_used.txt")

    with open(file_in, 'r') as ifile:
        for i, l in enumerate(ifile):
            count = i + 1
        print 'raw data lines: %d' % count
    with open(file_in, 'r') as ifile:
        with open(file_out, 'a') as ofile:
            for i in range(count):
                # print "%d of %d lines" % (i+1, count)
                l = ifile.readline()
                # if i == 1000:
                #     break
                if l[0] == '>':
                    location_search = re.search(r".+(\[)(?P<location>.+?)(\])$", l)
                    location = location_search.group('location').rstrip()
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
                            if location != 'NULL' and write_file != 0:
                                scores = getscores(score_d, aalist, seq)
                                ofile.write('%s|%s\n' % (location, scores))
                                entry_count += 1
                                print 'number of entries: %d' % entry_count
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
                        scores = getscores(score_d, aalist, seq)
                        ofile.write('%s|%s\n' % (location, scores))
                        entry_count += 1
                        print 'number of entries: %d' % entry_count
                        if outsize != 'all':
                            if entry_count == outsize:
                                break
                    del seq



def write_label_seq_file(file_in, file_out, write_file=0):
    count = 0
    with open(file_in, 'r') as ifile:
        for i, l in enumerate(ifile):
            count = i + 1
        print 'num lines: %d' % count
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
                    print location

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

        for k,v in d.iteritems():
            print "l: %s count:%d" % (k,v)


def check_label_seq_file_validity(filename):
    print "\nchecking validity of output file..."
    non_alpha_count = 0
    invalid_label_count = 0
    invalid_chars = ['[', ']', '-', ',', '.', '|', '\\']
    with open(filename, 'r') as ifile:

        for l in ifile:
            label, seq = l.strip().split('|')

            if not seq.isalpha():
                print "non alpha detected in seq!"
                non_alpha_count += 1
                for i in seq:
                    if not i.isalpha():
                        print i

            if any(c in label for c in invalid_chars):
                invalid_label_count += 1
                print label

        if non_alpha_count != 0 or invalid_label_count != 0:
            raise Exception("output file not valid")
        else:
            print "\noutput file seems fine\n"


def read_preprocessed_data(input_file, exclude_labels_less_than=0):
    """
    reads in label_scores.txt file and returns the labels and features as lists
    :param input_file: directory of label_scores.txt file
    :param exclude_labels_less_than: skip labels with occurrence less than this value
    :return: (labels, features)
    """
    labels = list()
    features = list()
    d = dict()

    with open(input_file, 'r') as ifile:
        for line in ifile:
            # print line
            temp = line.rstrip().split('|')
            try:
                d[temp[0]] += 1
            except KeyError:
                d[temp[0]] = 1

        ifile.seek(0)

        for line in ifile:

            temp = line.rstrip().split('|')

            if d[temp[0]] < exclude_labels_less_than:
                continue
            else:
                labels.append(temp[0])
                features.append(map(float, temp[1:]))

    return labels, features


if __name__ == '__main__':
    input_file = "../../data/plants/all_plants.fas_updated04152015"
    # input_file = '/Volumes/RAMDisk/all_plants.fas_updated04152015'
    output_file0 = "../../data/plants/label_seq.txt"
    output_file1 = "../../data/plants/label_scores.txt"
    # output_file1 = "/Volumes/RAMDisk/label_scores.txt"
    enable_write = 1

    # number of entries to output in the label & scores file.... max is 1257123
    size = 20000

    # UNCOMMENT THIS BLOCK TO OUTPUT LABEL & SEQUENCE file
    # if os.path.exists(output_file0) and enable_write != 0:
    #     os.remove(output_file0)
    # write_label_seq_file(input_file, output_file0, write_file=enable_write)
    # find_unique_labels(output_file0)
    # check_label_seq_file_validity(output_file0)

    # UNCOMMENT THIS BLOCK TO OUTPUT LABEL & SCORES file
    if os.path.exists(output_file1) and enable_write != 0:
        os.remove(output_file1)
    write_label_score_file(input_file, output_file1, write_file=enable_write, outsize=size)
    print('%s contains these labels:' % output_file1)
    find_unique_labels(output_file1)

import os
import re


def write_label_seq_file(file_in, file_out, write_file=0):
    count = 0
    with open(file_in, 'r') as ifile:
        for i, l in enumerate(ifile):
            count = i+1
        print 'num lines: %d' % count
    with open(file_in, 'r') as ifile:
        with open(file_out, 'a') as ofile:
            for i in range(count):
                # print 'in for...\n'
                l = ifile.readline()
                # if i == 1000:
                #     break
                if l[0] == '>':
                    # print 'in if...'

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
                    # print 'in else...'
                    seq = ''
                    seq += l.rstrip()
                    while True:
                        # print 'in while...'
                        x = ifile.tell()
                        l = ifile.readline()
                        if len(l) == 0:  # EOF
                            # do something
                            # print seq
                            if location != 'NULL' and write_file != 0:
                                ofile.write('%s|%s\n' % (location, seq))
                            del seq

                            return
                        if l[0] == '>':
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
            label, seq = l.strip().split('|')

            if label in d:
                d[label] += 1
            else:
                d[label] = 1

        for k,v in d.iteritems():
            print "k: %s v:%d" % (k,v)


def check_output_file_validity(filename):
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

if __name__ == '__main__':
    input_file = "../../data/plants/all_plants.fas_updated04152015"
    output_file = "../../data/plants/label_seq.txt"

    write_file = 0

    # if os.path.exists(output_file) and write_file != 0:
    #     os.remove(output_file)
    # write_label_seq_file(input_file, output_file, write_file=write_file)

    find_unique_labels(output_file)
    check_output_file_validity(output_file)
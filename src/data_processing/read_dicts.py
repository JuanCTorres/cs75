
def get_aaindex_list(filename):
    aalist = list()
    with open(filename, 'r') as ifile:
        for l in ifile:
            aalist.append(l.rstrip())
    return aalist


def add_to_correlation_dict(key, correlation_dict, raw_data):
    row = raw_data.rstrip().split()
    for i in range(0, len(row) - 1, 2):
        corr_id, corr = row[i].rstrip(), float(row[i + 1])
        # TODO is there a need to double key
        if key in correlation_dict:
            correlation_dict[key][corr_id] = corr
        else:
            inner_d = dict()
            inner_d[corr_id] = corr
            correlation_dict[key] = inner_d

        # print "k: %s v: (%s, %f)" % (key, tup[0], tup[1])


def add_to_score_dict(id, d, raw_data0, raw_data1, raw_data2):
    inner_d = dict()

    row0 = raw_data0.rstrip().split()
    row1 = raw_data1.rstrip().split()
    row2 = raw_data2.rstrip().split()

    if id in d:
        raise Exception("key % shouldn't exist in score_d yet" % id)

    for i in range(0, 10):
        a, b = row0[i].split('/')

        try:        # catch float('NA') error
            inner_d[a] = float(row1[i])
        except ValueError:
            inner_d[a] = 0

        try:
            inner_d[b] = float(row2[i])
        except ValueError:
            inner_d[b] = 0

    d[id] = inner_d


def construct_dicts(file_in):
    score_d = dict()
    corr_d = dict()

    with open(file_in, 'r') as ifile:
        for i, l in enumerate(ifile):
            count = i+1
        print 'aaindex lines: %d' % count

    with open(file_in, 'r') as ifile:
        for i in range(count):

            # TODO test set to break at 30
            # if i == 30:
            #     break

            l = ifile.readline()

            if l == '':     # EOF
                pass

            elif l[0] == 'H':     # ID of dict
                id = l.split(' ')[1].rstrip()
                # print id

            elif l[0] == 'C':     # correlations
                l = l[1:]
                add_to_correlation_dict(id, corr_d, l)

                while True:
                    x = ifile.tell()
                    l = ifile.readline()
                    if l[0] != ' ':
                        ifile.seek(x)
                        break
                    else:
                        add_to_correlation_dict(id, corr_d, l)

            elif l[0] == 'I':     # dict values
                row0 = l[1:]
                row1 = ifile.readline()
                row2 = ifile.readline()

                add_to_score_dict(id, score_d, row0, row1, row2)

            elif l[0] == '//':    # end of a dict
                pass

    return score_d, corr_d


if __name__ == '__main__':
    input_file0 = "../../data/aaindex/aaindex1.txt"
    input_file1 = "../../data/aaindex/aaindex_used.txt"

    d1, d2 = construct_dicts(input_file0)

    # for k, v in d2.iteritems():
    #     print "%s %s" % (k, str(v))
    #
    # for k, v in d1.iteritems():
    #     print "%s -> %s" % (k, str(v))

    print "len(d1): %d (should be 566)" % len(d1.keys())
    print "len(d2): %d" % len(d2.keys())

    l = get_aaindex_list(input_file1)
    print l


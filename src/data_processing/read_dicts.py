
def add_to_correlation_dict(key, correlation_dict, raw_data):
    row = raw_data.rstrip().split()
    for i in range(0, len(row) - 1, 2):
        tup = (row[i].rstrip(), float(row[i + 1]))

        if key in correlation_dict:
            correlation_dict[key].append(tup)
        else:
            correlation_dict[key] = [tup]

        print "k: %s v: (%s, %f)" % (key, tup[0], tup[1])


def add_to_dict_to_dict(id, d, raw_data0, raw_data1, raw_data2):
    inner_d = dict()

    row0 = raw_data0.rstrip().split()
    row1 = raw_data1.rstrip().split()
    row2 = raw_data2.rstrip().split()

    if id in d:
        raise Exception("key % shouldn't exist in d_to_d yet" % id)

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


def construct_dict(file_in):
    d_to_d = dict()
    corr_d = dict()

    with open(file_in, 'r') as ifile:
        for i, l in enumerate(ifile):
            count = i+1
        print 'num lines: %d' % count

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
                print id

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

                add_to_dict_to_dict(id, d_to_d, row0, row1, row2)

            elif l[0] == '//':    # end of a dict
                pass

    return d_to_d, corr_d


if __name__ == '__main__':
    input_file = "../../data/aaindex/aaindex1.txt"
    d1, d2 = construct_dict(input_file)

    for k, v in d2.iteritems():
        print "%s %s" % (k, str(v))

    for k, d in d1.iteritems():
        for aa, score in d.iteritems():
            print "%s -> %s" % (k, str((aa, score)))

    print "len(d1): %d (should be 566)" % len(d1)
    print "len(d2): %d" % len(d2)
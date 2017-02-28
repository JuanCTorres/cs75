from random import randint
import sys


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

        try:  # catch float('NA') error
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
            count = i + 1
        print('aaindex lines: %d' % count)

    with open(file_in, 'r') as ifile:
        for i in range(count):

            # TODO test set to break at 30
            # if i == 30:
            #     break

            l = ifile.readline()

            if l == '':  # EOF
                pass

            elif l[0] == 'H':  # ID of dict
                id = l.split(' ')[1].rstrip()
                # print id

            elif l[0] == 'C':  # correlations
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

            elif l[0] == 'I':  # dict values
                row0 = l[1:]
                row1 = ifile.readline()
                row2 = ifile.readline()

                add_to_score_dict(id, score_d, row0, row1, row2)

            elif l[0] == '//':  # end of a dict
                pass

    return score_d, corr_d


def select_dicts(dict, file_out):
    all_dicts, max_score, best_list = dict.keys(), 0, None
    for repeat in range(100):
        prev_list, dict_list, list_score, try_again = None, all_dicts[:], 0, 0
        while try_again < 1000:
            if prev_list == dict_list:
                try_again += 1
            else:
                try_again = 0
            prev_list, curr_dict = dict_list, dict[dict_list[randint(0, (len(dict_list) - 1))]]
            for element in curr_dict:
                if element in dict_list and curr_dict[element] > 0.8:
                    list_score += float(curr_dict[element])
                    dict_list.remove(element)
        list_score = list_score / (len(all_dicts) - len(dict_list))
        if list_score > max_score:
            best_list, max_score = dict_list[:], list_score

    print("Best List: ")
    print(best_list)
    print("length: " + str(len(best_list)) + "; score: " + str(max_score))
    out_file = open(file_out, 'w')
    for element in best_list:
        out_file.write("%s\n" % element)
    out_file.close()


def find_similar(d, threshold):
    sl = list()
    target = list()
    with open("../../data/aaindex/aaindex_search.txt", "r") as ifile:
        for l in ifile:
            # print l
            sl.append(l.rstrip())

    with open("../../data/aaindex/aaindex_target.txt", "r") as ifile:
        for l in ifile:
            # print l
            target.append(l.rstrip())
    all_related = set()
    for e in target:
        try:
            for k, v in d[e].items():
                print k
                print v
                if v >= threshold:
                    all_related.add(k)
        except KeyError:
            print('KE: %s' % k)
            pass

    print target
    print sl
    for t in target:
        all_related.add(t)

    overlap = list()
    # all_related.append(target)
    print all_related
    for i in sl:
        if i in all_related:
            overlap.append(i)

    print("%d items found that have >= %f correlation" % (len(overlap), threshold))
    print(overlap)

if __name__ == '__main__':

    input_file0 = "../../data/aaindex/aaindex1.txt"
    input_file1 = "../../data/aaindex/aaindex_used.txt"

    d1, d2 = construct_dicts(input_file0)

    # for k, v in d2.iteritems():
    #     print "%s %s" % (k, str(v))
    #
    # for k, v in d1.iteritems():
    #     print "%s -> %s" % (k, str(v))
    if len(sys.argv) > 1:
        select_dicts(d2, input_file1)
    print("len(d1): %d (should be 566)" % len(d1.keys()))
    print("len(d2): %d" % len(d2.keys()))

    # find_similar(d2, 0.93)

    l = get_aaindex_list(input_file1)


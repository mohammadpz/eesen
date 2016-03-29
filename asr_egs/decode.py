import numpy
import kaldi_io
from collections import defaultdict
from subprocess import call
floatX = 'float32'


ind_to_char = {
    0: '-',
    1: 'A',
    2: 'C',
    3: 'G',
    4: 'K',
    5: 'O',
    6: 'S',
    7: 'V',
    8: ' ',  # <SPACE>
    9: 'T',
    10: 'B',
    11: 'F',
    12: 'J',
    13: 'N',
    14: 'R',
    15: 'Y',
    16: '<NOISE>',
    17: "'",
    18: 'E',
    19: 'I',
    20: 'M',
    21: 'Q',
    22: 'U',
    23: 'X',
    24: '.',
    25: 'D',
    26: 'H',
    27: 'L',
    28: 'P',
    29: 'W',
    30: 'Z'}

softmax_to_fst = {
    0: 0,
    1: 14,
    2: 30,
    3: 32,
    4: 36,
    5: 40,
    6: 44,
    7: 48,
    8: 51,
    9: 3,
    10: 49,
    11: 31,
    12: 35,
    13: 39,
    14: 43,
    15: 47,
    16: 54,
    17: 1,
    18: 10,
    19: 34,
    20: 38,
    21: 42,
    22: 46,
    23: 50,
    24: 53,
    25: 15,
    26: 33,
    27: 37,
    28: 41,
    29: 45,
    30: 52,
    31: 55}


def make_wsj_probs(net_output, log_probs_ark_path):
    # a list of test examples, each with a shape of Time x 32
    softmax_log_probs, trans = numpy.load(net_output)

    fst_format_log_probs = [-10000 * numpy.ones((i.shape[0], 64))
                            for i in softmax_log_probs]

    for i in range(len(softmax_log_probs)):
        for col in range(32):
            new_col = softmax_to_fst[col]
            fst_format_log_probs[i][:, new_col] = \
                softmax_log_probs[i][:, col]

    log_probs_ark_path = 'ark:' + log_probs_ark_path
    writer = kaldi_io.BaseFloatMatrixWriter(log_probs_ark_path)
    for i, exp in enumerate(fst_format_log_probs):
        writer.write(str(i), exp)

    all_mapped_trans = []
    for i in range(len(trans)):
        new_trans = []
        for inx in trans[i]:
            new_trans.append(softmax_to_fst[inx + 1])
        all_mapped_trans.append(new_trans)

    print "Ark file created"

    return log_probs_ark_path, all_mapped_trans, trans


def ctc_strip(sequence, blank_symbol=0):
    res = []
    for i, s in enumerate(sequence):
        if (s != blank_symbol) and (i == 0 or s != sequence[i - 1]):
            res += [s]
    return numpy.asarray(res)


class Evaluation(object):
    @classmethod
    def levenshtein(cls, predicted_seq, target_seq, predicted_mask=None,
                    target_mask=None, eol_symbol=-1):
        """
        Informally, the Levenshtein distance between two
        sequences is the minimum number of symbol edits
        (i.e. insertions, deletions or substitutions) required to
        change one word into the other. (From Wikipedia)
        """
        if predicted_mask is None:
            plen, tlen = len(predicted_seq), len(target_seq)
        else:
            assert len(target_mask) == len(target_seq)
            assert len(predicted_mask) == len(predicted_seq)
            plen, tlen = int(sum(predicted_mask)), int(sum(target_mask))

        dist = [[0 for i in range(tlen + 1)] for x in range(plen + 1)]
        # dist = np.zeros((plen, tlen), dype=floatX)
        for i in xrange(plen + 1):
            dist[i][0] = i
        for j in xrange(tlen + 1):
            dist[0][j] = j

        for i in xrange(plen):
            for j in xrange(tlen):
                if predicted_seq[i] != target_seq[j]:
                    cost = 1
                else:
                    cost = 0
                dist[i + 1][j + 1] = min(dist[i][j + 1] + 1,   # deletion
                                         dist[i + 1][j] + 1,   # insertion
                                         dist[i][j] + cost)   # substitution

        return dist[-1][-1]

    @classmethod
    def wer(cls, predicted_seq, target_seq, predicted_mask=None,
            target_mask=None,
            eol_symbol=-1):
        """
        Word Error Rate is 'levenshtein distance' devided by
        the number of elements in the target sequence.
        Input may also be batches of data
        """
        if predicted_mask is None:
            error_rate = []
            for (single_predicted_seq, single_target_seq) in zip(predicted_seq,
                                                                 target_seq):

                l_dist = cls.levenshtein(predicted_seq=single_predicted_seq,
                                         target_seq=single_target_seq,
                                         eol_symbol=eol_symbol)
                error_rate += [l_dist / float(len(single_target_seq))]
        else:
            error_rate = []
            # iteration over columns
            for (single_predicted_seq, single_p_mask,
                 single_target_seq, single_t_mask) in zip(predicted_seq.T,
                                                          predicted_mask.T,
                                                          target_seq.T,
                                                          target_mask.T):

                l_dist = cls.levenshtein(predicted_seq=single_predicted_seq,
                                         target_seq=single_target_seq,
                                         predicted_mask=single_p_mask,
                                         target_mask=single_t_mask,
                                         eol_symbol=eol_symbol)
                error_rate += [l_dist / sum(single_t_mask)]
        # returns an array for every single example in the batch
        return numpy.array(error_rate)

ind_to_48_phones = {1: 'sil',
                    2: 'aa',
                    3: 'ae',
                    4: 'ah',
                    5: 'ao',
                    6: 'aw',
                    7: 'ax',
                    8: 'ay',
                    9: 'b',
                    10: 'ch',
                    11: 'cl',
                    12: 'd',
                    13: 'dh',
                    14: 'dx',
                    15: 'eh',
                    16: 'el',
                    17: 'en',
                    18: 'epi',
                    19: 'er',
                    20: 'ey',
                    21: 'f',
                    22: 'g',
                    23: 'hh',
                    24: 'ih',
                    25: 'ix',
                    26: 'iy',
                    27: 'jh',
                    28: 'k',
                    29: 'l',
                    30: 'm',
                    31: 'n',
                    32: 'ng',
                    33: 'ow',
                    34: 'oy',
                    35: 'p',
                    36: 'r',
                    37: 's',
                    38: 'sh',
                    39: 't',
                    40: 'th',
                    41: 'uh',
                    42: 'uw',
                    43: 'v',
                    44: 'vcl',
                    45: 'w',
                    46: 'y',
                    47: 'z',
                    48: 'zh'}

dict_48_to_39_phones = {
    'aa': 'aa',
    'ae': 'ae',
    'ah': 'ah',
    'ao': 'aa',
    'aw': 'aw',
    'ax': 'ah',
    'er': 'er',
    'ay': 'ay',
    'b': 'b',
    'vcl': 'sil',
    'ch': 'ch',
    'd': 'd',
    'dh': 'dh',
    'dx': 'dx',
    'eh': 'eh',
    'el': 'l',
    'm': 'm',
    'en': 'n',
    'ng': 'ng',
    'epi': 'sil',
    'er': 'er',
    'ey': 'ey',
    'f': 'f',
    'g': 'g',
    'sil': 'sil',
    'hh': 'hh',
    'hh': 'hh',
    'ih': 'ih',
    'ix': 'ih',
    'iy': 'iy',
    'jh': 'jh',
    'k': 'k',
    'cl': 'sil',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'ng': 'ng',
    'n': 'n',
    'ow': 'ow',
    'oy': 'oy',
    'p': 'p',
    'sil': 'sil',
    'cl': 'sil',
    'r': 'r',
    's': 's',
    'sh': 'sh',
    't': 't',
    'cl': 'sil',
    'th': 'th',
    'uh': 'uh',
    'uw': 'uw',
    'uw': 'uw',
    'v': 'v',
    'w': 'w',
    'y': 'y',
    'z': 'z',
    'zh': 'sh'}


def remove_duplicates(input_):
    output = []
    last = 'imunique'
    for i in input_:
        if i != last:
            output.append(i)
            last = i
    return output


def compute_wer_argmax(all_trans, probs_path):
    all_argmax_predict = []
    reader = kaldi_io.SequentialBaseFloatMatrixReader(probs_path)
    for name, value in reader:
        argmax_predict = ctc_strip(numpy.argmax(value, axis=1))
        argmax_predict = [i for i in argmax_predict]
        all_argmax_predict.append(argmax_predict)

    return (numpy.sum(Evaluation.wer(all_argmax_predict, all_trans)) /
            float(len(all_argmax_predict)))


def compute_wer(all_trans, predict_path):
    predict = open(predict_path, 'r').readlines()
    all_predict = []
    for line in predict:
        line = line.strip('\n').split()
        line = line[1:]
        predict = ctc_strip([int(el) for el in line])
        predict = [el for el in predict]
        all_predict.append(predict)

    return (numpy.sum(Evaluation.wer(all_predict, all_trans)) /
            float(len(all_predict)))

net_output = '/u/pezeshki/speech_project/net_output.npy'
log_probs_ark_path = '/u/pezeshki/eesen/asr_egs/files/probs.ark'
log_probs_ark_path, all_new_trans, all_old_trans = make_wsj_probs(
    net_output, log_probs_ark_path)

words_path = '/u/zhangy/lisa/eesen/asr_egs/wsj/data/lang_char_larger/words.txt'
words_file = open(words_path, 'r').readlines()
word_to_ind = {}
for line in words_file:
    line = line.strip('\n').split()
    word_to_ind[line[0]] = int(line[1])

trans_word_level = []
for example in all_old_trans:
    string = ''
    for ind in example:
        string = string + ind_to_char[ind]
    string = string.strip('\n').split()
    trans = []
    for word in string:
        trans.append(word_to_ind[word])
    trans_word_level.append(trans)


fst_path = '/u/zhangy/lisa/eesen/asr_egs/wsj/data/lang_char_larger_test_tgpr/TLG.fst'
tokens_path = '/u/zhangy/lisa/eesen/asr_egs/wsj/data/lang_char/tokens.txt'
save_path = '/u/pezeshki/eesen/asr_egs/files/res.ark'

acwts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 5, 8, 15, 30, 50, 100, 500]
beams = [1, 5, 10, 15, 17, 20, 30, 70, 150, 300, 700, 1000]
all_exps = []
i = 0
for acwt in acwts:
    for beam in beams:
        print 'EXPERIMENT:' + str(i)
        i = i + 1
        this_exp = {}
        call(['sh', '/u/pezeshki/eesen/asr_egs/script_moh.sh',
              str(beam), str(acwt), str(fst_path),
              str(log_probs_ark_path), str(save_path),
              str(tokens_path)])

        print "Beam search:"
        print str(compute_wer(trans_word_level, save_path))
        this_exp['acwt'] = acwt
        this_exp['beam'] = beam
        this_exp['wer'] = compute_wer(trans_word_level, save_path)
        print this_exp
        all_exps.append(this_exp)
numpy.save('all_exps', all_exps)

import numpy
from picklable_itertools import groupby
import kaldi_io
import subprocess
floatX = 'float32'


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


def compute_wer_argmax(trans_path, probs_path):
    trans = open(trans_path, 'r').readlines()
    all_trans = []
    for line in trans:
        trans = [int(el) for el in line.strip('\n').split()[1:]]
        trans = [ind_to_48_phones[ind] for ind in trans]
        trans = [dict_48_to_39_phones[phone] for phone in trans]
        all_trans.append(remove_duplicates(trans))

    all_argmax_predict = []
    reader = kaldi_io.SequentialBaseFloatMatrixReader(probs_path)
    for name, value in reader:
        argmax_predict = ctc_strip(numpy.argmax(value, axis=1))
        argmax_predict = [ind_to_48_phones[ind] for ind in argmax_predict]
        argmax_predict = [dict_48_to_39_phones[phone] for
                          phone in argmax_predict]
        all_argmax_predict.append(remove_duplicates(argmax_predict))

    return (numpy.sum(Evaluation.wer(all_argmax_predict, all_trans)) /
            float(len(all_argmax_predict)))


def compute_wer(trans_path, predict_path):
    trans = open(trans_path, 'r').readlines()
    all_trans = []
    for line in trans:
        trans = [int(el) for el in line.strip('\n').split()[1:]]
        trans = [ind_to_48_phones[ind] for ind in trans]
        trans = [dict_48_to_39_phones[phone] for phone in trans]
        all_trans.append(remove_duplicates(trans))

    predict = open(predict_path, 'r').readlines()
    all_predict = []
    for line in predict:
        line = line.strip('\n').split()[1:]
        predict = ctc_strip([int(el) for el in line])
        predict = [el - 1 for el in predict]
        predict = [ind_to_48_phones[ind] for ind in predict]
        predict = [dict_48_to_39_phones[phone] for phone in predict]
        all_predict.append(remove_duplicates(predict))

    return (numpy.sum(Evaluation.wer(all_predict, all_trans)) /
            float(len(all_predict)))

print "Best Path Decoding WER: " + str(compute_wer(
    trans_path='/u/pezeshki/eesen/asr_egs/files/trans.txt',
    predict_path='/u/pezeshki/eesen/asr_egs/files/res_bestpath.ark'))

print "LatGen Decoding WER: " + str(compute_wer(
    trans_path='/u/pezeshki/eesen/asr_egs/files/trans.txt',
    predict_path='/u/pezeshki/eesen/asr_egs/files/res_latgen.ark'))

print "Argmax Decoding WER: " + str(compute_wer_argmax(
    trans_path='/u/pezeshki/eesen/asr_egs/files/trans.txt',
    probs_path='ark:/u/pezeshki/eesen/asr_egs/files/probs.ark'))

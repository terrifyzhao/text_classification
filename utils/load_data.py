import pandas as pd
import os
from utils.data_utils import shuffle, pad_sequences


def load_char_vocab():
    path = os.path.join(os.path.dirname(__file__), '../input/vocab.txt')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


def seq_index(p_sentences):
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    for p_sentence in p_sentences:
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)

    p_list = pad_sequences(p_list, maxlen=15)

    return p_list


def load_data(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['sentence'].values[0:data_size]
    # h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, label = shuffle(p, label)

    p_index = seq_index(p)

    return p_index, label




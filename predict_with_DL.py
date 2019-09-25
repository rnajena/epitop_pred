from keras import models
from keras import layers
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

import numpy as np
# import tensorflow as tf
# # tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

class Protein_seq():
    def __init__(self, sequence, score, over_threshold, positions=None):
        self.sequence = sequence
        self.score = score
        self.over_threshold = over_threshold
        if positions == None:
            self.positions = list(range(1, len(self.sequence) + 1))
        else:
            self.positions = positions


def build_model(nodes, seq_length, dropout=0):
    model = models.Sequential()
    model.add(layers.Embedding(20, 10, input_length=seq_length))
    model.add(layers.Bidirectional(layers.LSTM(nodes, return_sequences=True, dropout=dropout, recurrent_dropout=0.2)))
    model.add(layers.Bidirectional(layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2)))
    model.add(layers.Dense(nodes))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


def load_raw_AA_data(path):
    X_test_old = pd.read_csv(path, delimiter='\t', dtype='str', header=None).values
    X_test = parse_amino(x=X_test_old, generator=False)

    return X_test


def parse_amino(x):
    amino = "GALMFWKQESPVICYHRNDT"
    encoder = LabelEncoder()
    encoder.fit(list(amino))
    out = []
    for i in x:
        dnaSeq = i[1].upper()
        encoded_X = encoder.transform(list(dnaSeq))
        out.append(encoded_X)
    return np.array(out)


def split_AA_seq(seq, slicesize):
    splited_AA_seqs = []
    for i in range(0, len(seq) - slicesize):
        splited_AA_seqs.append([i + (slicesize // 2), seq[i:i + 50]])

    return np.array(splited_AA_seqs)


if __name__ == "__main__":
    # dummy dict is the input
    # dummy_dict = {"name": "GALMFWKQESPVICYHRNDTGALMFWKQESPVICY"}
    dummy_dict = {"name": "GALMFWKQESPVICYHRNDTGALMFWKQESPVICYHRNDTGALMFWKQESPVICYHRNDTGALMFWKQESPVICYHRNDTGALMFWKQESPVDT"}
    cutoff = 0.5
    # slicesize is the amount of AA which are used as input to predict liklelyhood of epitope
    slicesize = 50
    nodes = slicesize

    model = build_model(nodes, seq_length=slicesize)
    # exit()
    # location of weights from the previously trained model
    model_path = "/home/go96bix/projects/epitop_pred/epitope_data/weights.best.loss.test_generator.hdf5"
    # load weights, after this step the model behaves as if we trained it
    model.load_weights(model_path)

    output_dict = {}

    # go over all entries in dict
    for file_name in dummy_dict.keys():
        # slice the long AA in short segments so it can be used as input for the neural network
        seq_slices = split_AA_seq(dummy_dict[file_name], slicesize)
        # parse input to numerical values
        X_test = parse_amino(seq_slices)
        # finally predict the epitopes
        Y_pred_test = model.predict(X_test)

        # the column 0 in Y_pred_test is the likelihood that the slice is NOT a Epitope, for us mostly interesting
        # is col 1 which contain the likelihood of being a epitope
        epi_score = Y_pred_test[:, 1]

        # use leading and ending zeros so that the score array has the same length as the input sequence
        score = np.zeros(len(dummy_dict[file_name]))
        # leading AAs which are not predictable get value of first predicted value (were this AA where involved)
        score[0:int(seq_slices[0][0])] = epi_score[0]
        # last AAs which are not predictable get value of last predicted value (were this AA where involved)
        score[int(seq_slices[-1][0]):]=epi_score[-1]
        score[np.array(seq_slices[:,0],dtype=int)] = epi_score

        score_bool = score > cutoff

        protein = Protein_seq(sequence=dummy_dict[file_name], score=score, over_threshold=score_bool)
        output_dict.update({file_name:protein})


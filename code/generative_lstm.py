# the goal here is to train a simple generative LSTM
# - this LSTM is operating at the word level, rather than the character level

import numpy as np
import pandas as pd
import collections
import re
import argparse

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Activation, Embedding, Dropout, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.utils import to_categorical


def get_data_by_author(author):

    assert author in ['EAP', 'HPL', 'MWS']

    train = pd.read_csv("../data/train.csv")
    
    # select the fields we want to keep
    train = train[train['author'] == author]
    train = train['text'].replace("\n", "<eos>").tolist()

    # simple regex to pad punctuation with whitespace. I do this so that we tokenize both the text and puncuation separately.
    train = [re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', s) for s in train]
    train = " ".join(train).split()

    return train


def build_vocab(data):

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def data_to_word_ids(data, word_to_id):
    return [word_to_id[word] for word in data if word in word_to_id]


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


def train_model():

    train = get_data_by_author('EAP')

    word_to_id = build_vocab(train)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    train_data = data_to_word_ids(train, word_to_id)

    num_steps = 30
    batch_size = 20
    train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

    hidden_size = 500

    model = Sequential()
    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation('softmax'))

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    print(model.summary())
    checkpointer = ModelCheckpoint(filepath='../models/model-{epoch:02d}.hdf5', verbose=1)
    num_epochs = 50

    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, callbacks=[checkpointer])
    
    model.save("../models/final_model.hdf5")


def generate_from_model():

    train = get_data_by_author('EAP')

    word_to_id = build_vocab(train)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    train_data = data_to_word_ids(train, word_to_id)

    model = load_model("../models/final_model.hdf5")

    num_steps = 30
    dummy_iters = 40

    # test data set
    test_data = train_data
    example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, vocabulary,
                                                     skip_step=1)

    num_sentences_to_predict = 10
    num_words_predict = 20

    results = []
    actuals = []
    for j in range(num_sentences_to_predict):
        true_print_out = "Actual words: "
        pred_print_out = "Predicted words: "

        for i in range(num_words_predict):
            data = next(example_test_generator.generate())
            prediction = model.predict(data[0])
            predict_word = np.argmax(prediction[:, num_steps - 1, :])
            
            true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
            pred_print_out += reversed_dictionary[predict_word] + " "

        actuals.append(true_print_out)
        results.append(pred_print_out)

        for _  in range(dummy_iters):
            dummy = next(example_test_generator.generate())
            dummy_iters += 1

    for i in range(len(actuals)):
        print (actuals[i])
        print (results[i], '\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--option', help='a string to govern funcationality: train or generate', required=True)
    args = vars(parser.parse_args())

    if args['option'] == 'train':
        train_model()
    elif args['option'] == 'generate':
        generate_from_model()

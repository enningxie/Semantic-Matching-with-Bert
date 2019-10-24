# coding=utf-8
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from keras.models import Model
from keras.layers import Lambda, Dense, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


class Bert(object):
    def __init__(self, mode='inference'):
        self.maxlen = 32
        self.config_path = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
        self.checkpoint_path = '/Data/public/Bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
        self.dict_path = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
        self.train_data_path = 'data/train_LCQMC.csv'
        self.dev_data_path = 'data/dev_LCQMC.csv'
        self.test_data_path = 'data/test_LCQMC.csv'
        self.restore_model_path = 'saved_models/bert_0801_1405.h5'
        self.token_dict = self._read_token_dict()
        self.tokenizer = self.OurTokenizer(self.token_dict)
        self.model = self._get_model()
        if mode == 'inference':
            self._init_model()

    # customize tokenizer
    class OurTokenizer(Tokenizer):
        def _tokenize(self, text):
            tokens = []
            for ch in text:
                if ch in self._token_dict:
                    tokens.append(ch)
                elif self._is_space(ch):
                    tokens.append('[unused1]')  # space类用未经训练的[unused1]表示
                else:
                    tokens.append('[UNK]')  # 剩余的字符是[UNK]
            return tokens

    # return char2index dict
    def _read_token_dict(self):
        token_dict = {}

        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        return token_dict

    def _seq_padding(self, X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        padded_sent = np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
        return padded_sent

    # bert for Semantic matching, model architecture
    def _get_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)

        for l in bert_model.layers:
            l.trainable = True

        sent_first = Input(shape=(None,))
        sent_second = Input(shape=(None,))

        x = bert_model([sent_first, sent_second])
        x = Lambda(lambda tmp_input: tmp_input[:, 0])(x)
        y_pred = Dense(1, activation='sigmoid')(x)

        model = Model([sent_first, sent_second], y_pred)
        return model

    # prepare data for training
    def _prepare_data(self, data_path):
        data = pd.read_csv(data_path)
        sent_1 = data['sentence1'].values
        sent_2 = data['sentence2'].values
        label = data['label'].values
        X1_pad, X2_pad = self._data_preprocessing(sent_1, sent_2)
        # X1 = np.vstack((X1_pad, X2_pad))
        # X2 = np.vstack((X2_pad, X1_pad))
        # y_train = np.hstack((label, label))
        return X1_pad, X2_pad, label

    # model training step
    def train(self):
        # train_data
        train_x1, train_x2, train_label = self._prepare_data(self.train_data_path)
        # dev_data
        dev_x1, dev_x2, dev_label = self._prepare_data(self.dev_data_path)
        checkpoint = ModelCheckpoint(self.restore_model_path, monitor='val_acc', verbose=0,
                                     save_best_only=True, save_weights_only=True)
        early_stop = EarlyStopping(monitor='val_acc', patience=4, verbose=0, mode='auto', baseline=None,
                                   restore_best_weights=True)
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        self.model.summary()
        self.model.fit(x=[train_x1, train_x2],
                       y=train_label,
                       batch_size=64,
                       epochs=10,
                       verbose=1,
                       callbacks=[checkpoint, early_stop],
                       validation_data=([dev_x1, dev_x2], dev_label))

    # data pre-processing operation
    def _data_preprocessing(self, sentence1, sentence2):
        X1, X2 = [], []
        for tmp_sent1, tmp_sent2 in zip(sentence1, sentence2):
            x1, x2 = self.tokenizer.encode(first=tmp_sent1[:self.maxlen], second=tmp_sent2[:self.maxlen])
            X1.append(x1)
            X2.append(x2)
        X1 = self._seq_padding(X1)
        X2 = self._seq_padding(X2)
        # X1 = pad_sequences(X1, maxlen=67, padding='post', truncating='post')
        # X2 = pad_sequences(X2, maxlen=67, padding='post', truncating='post')
        return X1, X2

    # model predict operation
    def predict(self, sentence1, sentence2):
        X1, X2 = self._data_preprocessing(sentence1, sentence2)
        y_pred = self.model.predict([X1, X2], batch_size=1024)
        return y_pred

    def test(self):
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy']
        )
        # test_data
        test_x1, test_x2, test_label = self._prepare_data(self.test_data_path)
        test_loss, test_acc = self.model.evaluate(x=[test_x1, test_x2], y=test_label)
        print('test loss: {}'.format(test_loss))
        print('test acc: {}'.format(test_acc))

    def _init_model(self):
        self.model.load_weights(self.restore_model_path)
        sentence1 = '干嘛呢'
        sentence2 = '你是机器人'
        print('model bert loaded succeed. ({})'.format(self.predict([sentence1], [sentence2]).item()))

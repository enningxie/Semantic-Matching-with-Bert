# coding=utf-8
from bert import Bert
from albert import Albert
import os
import numpy as np
import pandas as pd
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def exceed_threshold(y_pred, y_label, threshold=0.7):
    y_pred_np = np.asarray(y_pred)
    y_pred_len = len(y_pred_np)
    y_pred_label = (y_pred_np > threshold).astype('int')
    accuracy = np.sum(y_pred_label == y_label) / y_pred_len
    return accuracy


def evaluate_albert_tiny(csv_path, mode_, model_name, dataset_name):
    raw_df = pd.read_csv(csv_path)
    sent1_list = list(raw_df.sentence1)
    sent2_list = list(raw_df.sentence2)
    y_label = np.expand_dims(np.asarray(raw_df.label), -1)
    model_albert = Albert(mode='inference',
                          mode_=mode_,
                          model_name=model_name,
                          dataset_name=dataset_name)
    start_time = time.time()
    y_pred = model_albert.predict(sent1_list, sent2_list)
    duration = time.time() - start_time
    return duration, exceed_threshold(y_pred, y_label)


def run():
    # Bert
    # model_bert = Bert(mode='inference')
    # training
    # model_bert.train()

    # Albert
    model_albert = Albert(mode='train', mode_='part')
    # print(model_albert.predict(['我今天不买', '今天天气不错', '我不想买了', '很好', '不好', '没有啊', '买好了', '不需要'], ['我今天买好了', '今天天气不好','我想买', '不好', '不怎么样', '有啊', '已经买了', '没必要']))
    # while True:
    #     sent1 = input('sent1: ')
    #     sent2 = input('sent2: ')
    #     print(model_albert.predict([sent1], [sent2]).item())


if __name__ == '__main__':
    # run()
    tmp_models_config = [
        ['full', 'albert_tiny_250k.h5', 'LCQMC'],
        ['full', 'albert_tiny_250k_BQ.h5', 'LCQMC'],
        ['full', 'albert_tiny_250k.h5', 'LCQMC']
    ]
    models_config = [
        ['part', '01', 'LCQMC'],
        ['full', '02', 'LCQMC'],
        ['part', '03', 'BQ'],
        ['full', '04', 'BQ'],
        ['part', '05', 'sent_pair'],
        ['full', '06', 'sent_pair'],
        ['full', '07', 'LCQMC'],
        ['full', '08', 'LCQMC'],
        ['full', '09', 'LCQMC'],
        ['part', '10', 'sent_pair']
    ]
    durations = []
    accs = []
    for tmp_model_config in tmp_models_config:
        print('--------albert_tiny_{}----------'.format(tmp_model_config[1]))
        duration, acc = evaluate_albert_tiny('data/test_data(1).csv', *tmp_model_config)
        durations.append(duration)
        accs.append(acc)

    for model_index, (tmp_duration, tmp_acc) in enumerate(zip(durations, accs)):
        print('--------albert_tiny_{}----------'.format(model_index + 1))
        print('cost time: {}'.format(tmp_duration))
        print('acc: {}'.format(tmp_acc))
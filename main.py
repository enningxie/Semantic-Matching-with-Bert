# coding=utf-8
from bert import Bert
from albert import Albert
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def run():
    # Bert
    # model_bert = Bert(mode='inference')
    # training
    # model_bert.train()

    # Albert
    model_albert = Albert(mode='train', mode_='part')
    # model_albert.test()
    # while True:
    #     sent1 = input('sent1: ')
    #     sent2 = input('sent2: ')
    #     print(model_albert.predict([sent1], [sent2]).item())


if __name__ == '__main__':
    run()

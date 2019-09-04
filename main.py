# coding=utf-8
from bert import Bert
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def run():
    model_bert = Bert()
    # training
    model_bert.train()


if __name__ == '__main__':
    run()

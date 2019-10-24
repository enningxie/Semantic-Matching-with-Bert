from bert4keras.utils import SimpleTokenizer, load_vocab


if __name__ == '__main__':
    _token_dict = load_vocab('/Data/public/Bert/albert_tiny_250k/vocab.txt')  # 读取字典

    print(type(_token_dict))
    print(_token_dict)
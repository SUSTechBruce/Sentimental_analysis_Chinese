"""
Load the corpus from train/validation/test set
"""
from collections import Counter
from Sentimental_analysis_Chinese.Pre_process import category_to_id, word_to_id
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch


def load_corpus(path, word_to_id, max_sen_len=50):
    """

    # :param path: txt path
    # :return: contents and labels
    # """
    _, cat_to_id = category_to_id()
    contents = []
    labels = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]
            content = [word_to_id.get(w, 0) for w in sp[1:]]
            content = content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [word_to_id['_PAD_']] * (max_sen_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)
    print('Sum of samples = {}'.format(len(labels)))
    print('Num of categories')
    for w in counter:
        print(w, counter[w])
    contents = np.asarray(contents)
    labels = np.array([cat_to_id[i] for i in labels])

    return contents, labels


def load_train_data(train_contents, val_contents, train_labels, val_labels, batchsize):
    contents = np.vstack([train_contents, val_contents])
    labels = np.concatenate([train_labels, val_labels])

    # 加载训练用的数据
    train_dataset = TensorDataset(torch.from_numpy(contents).type(torch.float),
                                  torch.from_numpy(labels).type(torch.long))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batchsize,
                                  shuffle=True, num_workers=0)

    return train_dataloader


def load_test_data(test_contents, test_labels, batchsize):
    test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.float),
                                 torch.from_numpy(test_labels).type(torch.long))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batchsize,
                                 shuffle=False, num_workers=0)
    return test_dataloader


if __name__ == '__main__':
    word_to_id = word_to_id('./Dataset/word_to_id.txt')
    print('train corpus load: ')
    train_contents, train_labels = load_corpus('./Dataset/train.txt', word_to_id, max_sen_len=50)  # [idx of one stentence]
    print(train_contents, train_labels)
    print("train_contents shape", train_contents.shape)
    print("train_labels shape", train_labels.shape)
    print('validation corpus load: ')
    val_contents, val_labels = load_corpus('./Dataset/validation.txt', word_to_id, max_sen_len=50)
    print('test corpus load:')
    test_contents, test_labels = load_corpus('./Dataset/test.txt', word_to_id, max_sen_len=50)







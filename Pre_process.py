import gensim
import numpy as np


def word_to_id(file, save_to_path=None):
    """
    :param file: the file address of the train and test dataset
    :param save_to_path:
    :return: None
    """
    word_to_id = {'_PAD_': 0}
    path = ['./Dataset/train.txt', './Dataset/validation.txt']

    # open and store train.txt
    with open(path[0], encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            for character in sp[1:]:
                if character not in word_to_id.keys():
                    word_to_id[character] = len(word_to_id)
    # open and store validation.txt
    with open(path[1], encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            for character in sp[1:]:
                if character not in word_to_id.keys():
                    word_to_id[character] = len(word_to_id)
    if save_to_path:
        with open(file, 'w', encoding='utf-8') as f:
            for w in word_to_id:
                f.write(w + '\t')
                f.write(str(word_to_id[w]))
                f.write('\n')
    return word_to_id


def word_to_vector(fname, word2id, save_to_path=None):
    """
    :param frame: pre-trained word_to_vector from wiki
    :param word_to_id: index of words in dict
    :param save_to_path: save the result of word with the respect to word2vec
    :return:word_to_vector {id: word2vec}
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs


def category_to_id(classes=None):
    """
    :param classes: 0:positive, 1:negative
    :return: class id
    """
    if not classes:
        classes = ['0', '1']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    return classes, cat2id


if __name__ == '__main__':
    word_to_id = word_to_id('./Dataset/word_to_id.txt')
    print(word_to_id)
    word_to_vec = word_to_vector('./Dataset/wiki_word2vec_50.bin', word_to_id)
    assert word_to_vec.shape == (58954, 50)
    print(word_to_vec)
    print(word_to_vec.shape)

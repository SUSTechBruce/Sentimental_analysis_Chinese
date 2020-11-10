import torch
import torch.nn as nn
import torch.nn.functional as F
from Sentimental_analysis_Chinese.CN_segmentation import Thu_seg
import re
from Sentimental_analysis_Chinese.Pre_process import word_to_id, word_to_vector
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

word_to_id = word_to_id('./Dataset/word_to_id.txt')
word2vec = word_to_vector('./Dataset/wiki_word2vec_50.bin', word_to_id)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TextCNN(nn.Module):

    def __init__(self, CONFIG):
        super(TextCNN, self).__init__()
        self.update_w2v = CONFIG['update_w2v']
        self.vocab_size = CONFIG['vocab_size']
        self.n_class = CONFIG['n_class']
        self.embedding_dim = CONFIG['embedding_dim']
        self.drop_keep_prob = CONFIG['drop_keep_prob']
        self.num_filters = CONFIG['num_filters']
        self.kernal_size = CONFIG['kernel_size']
        self.pretrained_embed = CONFIG['pretrained_embed']

        # use pretrain word2vector
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)  # 58954 * 50
        torch_embedding = torch.from_numpy(self.pretrained_embed)
        self.embedding.weight.data.copy_(torch_embedding)
        self.embedding.weight.requires_grad = self.update_w2v
        # conV layer
        self.conv = nn.Conv2d(1, self.num_filters, (self.kernal_size, self.embedding_dim))
        # Dropout
        self.dropout = nn.Dropout(self.drop_keep_prob)
        # full connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.num_filters, self.n_class),
        )

    def forward(self, x):
        """

        X shape torch.Size([32, 50])
        Embedding x shape torch.Size([32, 50, 50])
        x.unsqueeze(1) torch.Size([32, 1, 50, 50])
        F.relu(self.conv(x)) torch.Size([32, 256, 48, 1])
        F.relu(self.conv(x)).squeeze(3) torch.Size([32, 256, 48])
        x = F.max_pool1d(x, x.size(2)) torch.Size([32, 256, 1])
        F.max_pool1d(x, x.size(2)).squeeze(2) torch.Size([32, 256])
        Dropout torch.Size([32, 256])
        fc torch.Size([32, 2])
        """
        x = x.to(torch.int64)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = F.relu(self.conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, lable_size, use_gpu, batch_size, pretrained_embed):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.pretrained_embed = pretrained_embed
        self.n_classes = lable_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)  # 58954 * 50
        torch_embedding = torch.from_numpy(self.pretrained_embed)
        self.embedding.weight.data.copy_(torch_embedding)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)  # 50, 256
        self.linear = nn.Linear(hidden_size, self.n_classes)
        self.linear_2 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # print("lstm_out", lstm_out.shape)
        x = self.linear(lstm_out)

        x = x.view(x.size(0), -1)
        x = self.linear_2(x)

        return x


class Bi_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, n_layers, vocab_size, output_size, use_gpu, batch_size,
                 pretrained_embed, drop_prob=0.5):
        super(Bi_LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.pretrained_embed = pretrained_embed
        self.n_layers = n_layers  # the layers nums of Bi-LSTM
        self.dropout = drop_prob

        # use pretrained embedding model to speed up the training process
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        torch_embedding = torch.from_numpy(self.pretrained_embed)
        self.embedding.weight.data.copy_(torch_embedding)
        self.embedding.weight.requires_grad = True

        self.Bi_lstm = nn.LSTM(embedding_dim, hidden_size, n_layers,
                               dropout=self.dropout, batch_first=True, bidirectional=True)
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size * 4, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        """
       Create two new tensors: (n_layers, batch_size, hidden_dims) 2 * 32 * 256
        """
        weight = next(self.parameters()).data
        # bi-directional
        hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_size).zero_().cuda(),
                  weight.new(self.n_layers * 2, batch_size, self.hidden_size).zero_().cuda())
        return hidden

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(torch.int64).to(device)
        x = self.embedding(x)  # 32 50 50
        lstm_out, hidden = self.Bi_lstm(x)  # lstm out = 32 50 256
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size * 4)
        # print("LSTM out shape", lstm_out.shape)
        out = self.dropout(lstm_out)
        out = self.linear(out)
        out = self.sigmoid(out)
        out = out.view(batch_size, -1)
        out_label = out[:, -1]
        return out_label, hidden


def get_sentence_infer(sentence):
    words = Thu_seg(sentence)
    reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
    for word in words:
        if re.search(reg, word):
            words.remove(word)

    print(words)
    max_length = 50
    content = [word_to_id.get(w, 0) for w in words]
    content = content[:max_length]
    if len(content) < max_length:
        content += [word_to_id['_PAD_']] * (max_length - len(content))
    print(content)
    tensor = torch.from_numpy(np.asarray(content))
    print(tensor.unsqueeze(0).shape)
    tensor = tensor.unsqueeze(0)
    return tensor


def predict_sentence(tensor, model):
    data = TensorDataset(tensor.type(torch.float))
    batch_size = 1
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, num_workers=0)
    if model == 'TextCNN':
        textcnn_pred(data_loader)
    elif model == 'LSTM':
        lstm_pred(data_loader)
    elif model == 'Bi_LSTM':
        bi_lstm_pred(data_loader)
    else:
        print("The model doesn't exist...")


def textcnn_pred(data_loader):
    CONFIG = {
        "update_w2v": True,
        "vocab_size": 58954,
        "n_class": 2,
        "embedding_dim": 50,
        "drop_keep_prob": 0.5,
        "num_filters": 256,
        "kernel_size": 3,
        "pretrained_embed": word2vec
    }
    model = TextCNN(CONFIG)
    model.load_state_dict(torch.load('./Save_model/TextCNN.pth'))
    print('loading the model successfully')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, test_x in enumerate(data_loader):
            test_X = test_x[0].to(device)
            output = model(test_X)
            result = output.argmax(dim=1)
            if result == 0:
                print("The movie is positive...")
            else:
                print("The movie is negative...")


def lstm_pred(data_loader):
    CONFIG_LSTM = {
        'embedding_dim': 50,
        'hidden_size': 256,
        'vocab_size': 58954,
        'lable_size': 2,
        'use_gpu': True,
        'batch_size': 32,
        'pretrained_embed': word2vec

    }
    model = LSTM(CONFIG_LSTM['embedding_dim'], CONFIG_LSTM['hidden_size'], CONFIG_LSTM['vocab_size'],
                 CONFIG_LSTM['lable_size'], CONFIG_LSTM['use_gpu'], 1,
                 CONFIG_LSTM['pretrained_embed'])
    model.load_state_dict(torch.load('./Save_model/model_LSTM_final.pth'))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, test_X in enumerate(data_loader):
            test_X = test_X[0].to(device)
            output = model(test_X)
            result = output.argmax(dim=1)
            if result == 0:
                print("The movie is positive...")
            else:
                print("The movie is negative...")


def bi_lstm_pred(data_loader):
    CONFIG_Bi_LSTM = {
        'embedding_dim': 50,
        'hidden_size': 256,
        'n_layers': 4,
        'vocab_size': 58954,
        'output_size': 2,
        'use_gpu': True,
        'batch_size': 32,
        'pretrained_embed': word2vec

    }

    model = Bi_LSTM(CONFIG_Bi_LSTM['embedding_dim'], CONFIG_Bi_LSTM['hidden_size'], CONFIG_Bi_LSTM['n_layers'],
                    CONFIG_Bi_LSTM['vocab_size'], CONFIG_Bi_LSTM['output_size'], CONFIG_Bi_LSTM['use_gpu'],
                    1,
                    CONFIG_Bi_LSTM['pretrained_embed'])
    model.load_state_dict(torch.load('./Save_model/model_LSTM2.pth'))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, test_X in enumerate(data_loader):
            test_X = test_X[0].to(device)
            output, h = model(test_X)
            pred = torch.round(output.squeeze())
            if pred.item() == 0:
                print("The movie is positive...")
            else:
                print("The movie is negative..")


if __name__ == "__main__":
    sentence_negative = '这部电影演的太烂的，演员也烂，台词也烂，整个电影就是一个烂片'
    sentence_positive = '这部电影有点善良。'
    tensor = get_sentence_infer(sentence_positive)
    predict_sentence(tensor, 'Bi_LSTM')

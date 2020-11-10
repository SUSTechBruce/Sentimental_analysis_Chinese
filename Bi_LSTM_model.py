import torch
import torch.nn as nn
from Sentimental_analysis_Chinese.Pre_process import word_to_vector, word_to_id
from Sentimental_analysis_Chinese.Load_data import load_corpus, load_train_data, load_test_data
import numpy as np

word_to_id = word_to_id('./Dataset/word_to_id.txt')
word2vec = word_to_vector('./Dataset/wiki_word2vec_50.bin', word_to_id)

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
# self, embedding_dim, hidden_size, n_layers, vocab_size, output_size, use_gpu, batch_size,
#                  pretrained_embed, drop_prob=0.5
CONFIG_Bi_LSTM = {
    'embedding_dim': 50,
    'hidden_size': 256,
    'n_layers': 2,
    'vocab_size': 58954,
    'output_size': 2,
    'use_gpu': True,
    'batch_size': 32,
    'pretrained_embed': word2vec

}

TRAIN_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "model_path": None,
    "verbose": True
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.linear = nn.Linear(hidden_size * 2, self.output_size)
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
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size * 2)
        # print("LSTM out shape", lstm_out.shape)
        out = self.dropout(lstm_out)
        out = self.linear(out)
        out = self.sigmoid(out)
        out = out.view(batch_size, -1)
        out_label = out[:, -1]
        return out_label, hidden


def train(dataloader):
    model = Bi_LSTM(CONFIG_Bi_LSTM['embedding_dim'], CONFIG_Bi_LSTM['hidden_size'], CONFIG_Bi_LSTM['n_layers'],
                    CONFIG_Bi_LSTM['vocab_size'], CONFIG_Bi_LSTM['output_size'], CONFIG_Bi_LSTM['use_gpu'],
                    CONFIG_Bi_LSTM['batch_size'],
                    CONFIG_Bi_LSTM['pretrained_embed'])
    # if model_path:
    #     model.load_state_dict(torch.load(model_path))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    criterion = nn.BCELoss()
    model.train()
    num_correct = 0
    batch_n = 0
    for epoch in range(TRAIN_CONFIG['epochs']):
        # hidden = model.init_hidden(TRAIN_CONFIG['batch_size'])
        for batch_id, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            # h = tuple([each.data for each in hidden])
            output, _ = model(X)
            loss = criterion(output.squeeze(), Y.float())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            # calculate the accuracy rate
            pred = torch.round(output.squeeze())
            correct_tensor = pred.eq(Y.float().view_as(pred))
            correct = correct_tensor.cpu().numpy()
            num_correct += np.sum(correct)
            batch_n += X.shape[0]

            if batch_id % 200 == 0 & TRAIN_CONFIG['verbose']:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {:.6f}'.format(
                    epoch + 1, batch_id * len(X), len(dataloader.dataset),
                    100. * batch_id / len(dataloader), loss.item(), num_correct / batch_n))
                num_correct = 0
                batch_n = 0

    torch.save(model.state_dict(), './Save_model/model_BiLSTM2_layer.pth')


def test_model(test_loader, model_path):
    model = Bi_LSTM(CONFIG_Bi_LSTM['embedding_dim'], CONFIG_Bi_LSTM['hidden_size'], CONFIG_Bi_LSTM['n_layers'],
                    CONFIG_Bi_LSTM['vocab_size'], CONFIG_Bi_LSTM['output_size'], CONFIG_Bi_LSTM['use_gpu'],
                    CONFIG_Bi_LSTM['batch_size'],
                    CONFIG_Bi_LSTM['pretrained_embed'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    num_correct = 0
    count = 0
    # h = model.init_hidden(CONFIG_Bi_LSTM['batch_size'])
    with torch.no_grad():
        for i, (test_X, test_Y) in enumerate(test_loader):
            # h = tuple([each.data for each in h])
            test_X = test_X.to(device)
            test_Y = test_Y.to(device)
            output, h = model(test_X)

            # calculate the accuracy rate
            pred = torch.round(output.squeeze())
            correct_tensor = pred.eq(test_Y.float().view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
            count += test_Y.shape[0]

    print('The test accuracy is {:.2f}%'.format(100 * num_correct / count))


if __name__ == '__main__':
    # Train the model
    print(device)
    train_contents, train_labels = load_corpus('./Dataset/train.txt', word_to_id, max_sen_len=50)
    val_contents, val_labels = load_corpus('./Dataset/validation.txt', word_to_id, max_sen_len=50)
    data_loader = load_train_data(train_contents, val_contents, train_labels, val_labels, TRAIN_CONFIG['batch_size'])
    train(data_loader)

    # Test the model
    test_contents, test_labels = load_corpus('./Dataset/test.txt', word_to_id, max_sen_len=50)
    test_loader = load_test_data(test_contents, test_labels, TRAIN_CONFIG['batch_size'])
    model_path = './Save_model/model_BiLSTM2_layer.pth'
    test_model(test_loader, model_path)

    # model tensor test
    # self, embedding_dim, hidden_size, n_layers, vocab_size, lable_size, use_gpu, batch_size,
    #                  pretrained_embed, drop_prob=0.5)
    # model = Bi_LSTM(50, 256, 2, 58954, 1, True, 32, word2vec).to(device)
    # x = torch.arange(0, 32 * 50).reshape(32, 50).to(device)
    # h = model.init_hidden(32)
    # print(x)
    # x, _ = model(x, h)
    # print(x.shape)
    # print(model)

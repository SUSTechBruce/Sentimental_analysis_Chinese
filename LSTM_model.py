import torch
import torch.nn as nn
import torch.nn.functional as F
from Sentimental_analysis_Chinese.Pre_process import word_to_vector, word_to_id
from torchsummary import summary
from Sentimental_analysis_Chinese.Load_data import load_corpus, load_train_data, load_test_data

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
CONFIG_LSTM = {
    'embedding_dim': 50,
    'hidden_size': 256,
    'vocab_size': 58954,
    'lable_size': 2,
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


def train(dataloader):
    model = LSTM(CONFIG_LSTM['embedding_dim'], CONFIG_LSTM['hidden_size'], CONFIG_LSTM['vocab_size'],
                 CONFIG_LSTM['lable_size'], CONFIG_LSTM['use_gpu'], CONFIG_LSTM['batch_size'],
                 CONFIG_LSTM['pretrained_embed'])
    # if model_path:
    #     model.load_state_dict(torch.load(model_path))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    model.train()
    accuracy = 0
    batch_n = 0
    for epoch in range(TRAIN_CONFIG['epochs']):
        for batch_id, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss = criterion(output, Y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            accuracy += (output.argmax(dim=1) == Y).sum().to(torch.float).item()
            batch_n += X.shape[0]
            if batch_id % 200 == 0 & TRAIN_CONFIG['verbose']:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {:.6f}'.format(
                    epoch + 1, batch_id * len(X), len(dataloader.dataset),
                    100. * batch_id / len(dataloader), loss.item(), accuracy / batch_n))
                accuracy = 0
                batch_n = 0

    torch.save(model.state_dict(), './Save_model/model_LSTM_final.pth')


def test_model(test_loader, model_path):
    model = LSTM(CONFIG_LSTM['embedding_dim'], CONFIG_LSTM['hidden_size'], CONFIG_LSTM['vocab_size'],
                 CONFIG_LSTM['lable_size'], CONFIG_LSTM['use_gpu'], CONFIG_LSTM['batch_size'],
                 CONFIG_LSTM['pretrained_embed'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    count = 0
    correct = 0
    with torch.no_grad():
        for i, (test_X, test_Y) in enumerate(test_loader):
            test_X = test_X.to(device)
            test_Y = test_Y.to(device)
            output = model(test_X)
            correct += (output.argmax(dim=1) == test_Y.long()).sum().item()
            count += len(test_X)

    print('The test accuracy is {:.2f}%'.format(100 * correct / count))


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
    model_path = './Save_model/model_LSTM_final.pth'
    test_model(test_loader, model_path)
    # model = LSTM(50, 256, 58954, 2, False, 32, word2vec)
    # model2 = TextCNN(CONFIG)
    # x = torch.arange(0, 32 * 50).reshape(32, 50)
    # print(x)
    # x = model(x)
    # print(x.shape)

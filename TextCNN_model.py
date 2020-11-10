import torch
import torch.nn as nn
import torch.nn.functional as F
from Sentimental_analysis_Chinese.Pre_process import word_to_vector, word_to_id
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

TRAIN_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "model_path": None,
    "verbose": True
}

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


def train(dataloader):
    model = TextCNN(CONFIG)
    # if model_path:
    #     model.load_state_dict(torch.load(model_path))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(TRAIN_CONFIG['epochs']):
        for batch_id, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss = criterion(output, Y)

            if batch_id % 200 == 0 & TRAIN_CONFIG['verbose']:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_id * len(X), len(dataloader.dataset),
                    100. * batch_id / len(dataloader), loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), './Save_model/TextCNN.pth')


def test_model(test_loader, model_path):
    model = TextCNN(CONFIG)
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
    # # Train the model
    # print(device)
    # train_contents, train_labels = load_corpus('./Dataset/train.txt', word_to_id, max_sen_len=50)
    # val_contents, val_labels = load_corpus('./Dataset/validation.txt', word_to_id, max_sen_len=50)
    # data_loader = load_train_data(train_contents, val_contents, train_labels, val_labels, TRAIN_CONFIG['batch_size'])
    # train(data_loader)
    #
    # # Test the model
    # test_contents, test_labels = load_corpus('./Dataset/test.txt', word_to_id, max_sen_len=50)
    # test_loader = load_test_data(test_contents, test_labels, TRAIN_CONFIG['batch_size'])
    # model_path = './Save_model/TextCNN.pth'
    # test_model(test_loader, model_path)
    print(torch.__version__)




import random
import copy
import torch
import numpy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import argparse


# samples_size = 5000
# test_size = 0.2
# val_size = 0.1
LEARNING_RATE = 0.05
# NUM_EPOCHS = 20
BATCH_SIZE = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def create_sequence(size=1000):
    X = torch.randint(low=0, high=10, size=(size, 1))
    Y = torch.clone(X)
    Y[1:] = (X[1:] + X[0]) % 10
    return X, Y


class RNNAll(torch.nn.Module):

    def __init__(self, size_of_dict=11, emded_dim=10, hidden_size=10, out_size=10, hiddenRNN=torch.nn.RNN):
        super(RNNAll, self).__init__()
        self.embed = torch.nn.Embedding(size_of_dict, emded_dim)
        self.hidden = hiddenRNN(emded_dim, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, out_size)

    def forward(self, sequence):
        embed = self.embed(sequence)
        o, _ = self.hidden(embed)
        return self.linear(o)


def train(NUM_EPOCHS=20, samples_size=1000, test_size=0.2, val_size=0.1):
    X_samples, Y_samples = create_sequence(size=samples_size)
    X_train, X_val, y_train, y_val = train_test_split(X_samples, Y_samples, test_size=val_size, stratify=Y_samples)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)

    traindataset = torch.utils.data.TensorDataset(X_train, y_train)
    traindata = torch.utils.data.DataLoader(traindataset, BATCH_SIZE, shuffle=True)

    testdataset = torch.utils.data.TensorDataset(X_test, y_test)
    testdata = torch.utils.data.DataLoader(testdataset, BATCH_SIZE, shuffle=True)

    valdataset = torch.utils.data.TensorDataset(X_val, y_val)
    valdata = torch.utils.data.DataLoader(valdataset, BATCH_SIZE, shuffle=True)

    hidden_size = 200
    out_size = 10

    model_dict = {'RNN': torch.nn.RNN,
                  'GRU': torch.nn.GRU,
                  'LSTM': torch.nn.LSTM}
    print(f'First element start sequence {X_samples[0]}. Numbers of samples {samples_size}.'
          f' Numbers of epoch {NUM_EPOCHS}.')
    print('-' * 20)
    for name in model_dict:
        print(f'Traning {name} recurrent network. Wait please...')
        model = RNNAll(size_of_dict=20, hidden_size=hidden_size, out_size=out_size, hiddenRNN=model_dict[name]).to(
            DEVICE)
        loss = torch.nn.CrossEntropyLoss().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc, iter_num = .0, .0, .0
            model.train()
            for x_in, y_in in traindata:
                x_in = x_in.to(DEVICE)
                y_in = y_in.to(DEVICE)
                y_in = y_in.view(1, -1).squeeze()
                optimizer.zero_grad()
                out = model.forward(x_in)
                out = out.view(-1, out_size).squeeze()
                l = loss(out, y_in)
                train_loss += l.item()
                batch_acc = (out.argmax(dim=1) == y_in)
                train_acc += batch_acc.sum().item() / batch_acc.shape[0]
                l.backward()
                optimizer.step()
                iter_num += 1

            test_loss, test_acc, iter_num = .0, .0, .0
            model.eval()
            for x_in, y_in in testdata:
                x_in = x_in.to(DEVICE)
                y_in = y_in.view(1, -1).squeeze().to(DEVICE)
                out = model.forward(x_in).view(-1, out_size).squeeze()
                l = loss(out, y_in)
                test_loss += l.item()
                batch_acc = (out.argmax(dim=1) == y_in)
                test_acc += batch_acc.sum().item() / batch_acc.shape[0]
                iter_num += 1
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)

        val_results = model(X_val.to(DEVICE)).argmax(dim=2)
        val_acc = (val_results == y_val.to(DEVICE)).flatten()
        val_acc = (val_acc.sum() / val_acc.shape[0]).item()
        print(f'For {name} recurrent network')
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"X validation sequence is: \"{X_val[:20].view(1, -1)}\"")
        print(f"Y validation sequence: \"{y_val[:20].view(1, -1)}\"")
        print(f"Predict sequence: \"{val_results[:20].view(1, -1)}\"")
        print('-' * 20)


parse = argparse.ArgumentParser()
parse.add_argument('--size', '-s', help='Sample size', default=3000, type=int)
parse.add_argument('--epoch', '-e', help='Numbers of epoch', default=20, type=int)
parse.add_argument('--val', '-v', help='Size of validation sample', default=0.1, type=float)
parse.add_argument('--test', '-t', help='Size of test sample', default=0.2, type=float)
args = parse.parse_args()

args = parse.parse_args()
NUM_EPOCHS = args.epoch
samples_size = args.size
test_size = args.test
val_size = args.val

train(NUM_EPOCHS=NUM_EPOCHS, samples_size=samples_size, test_size=test_size, val_size=val_size)
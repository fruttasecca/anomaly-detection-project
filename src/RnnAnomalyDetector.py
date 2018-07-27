import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.stats import multivariate_normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        encodes input window into latent space --> vector containing [mu sigma]
        :param input_size: window size of input
        :param hidden_size: size of single hidden layer
        :param num_layers: number of hidden layers
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.hidden = self.init_hidden()

    def forward(self, input):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        output = self.linear(input)
        output = output.view(1, 1, -1)
        output, self.hidden = self.lstm(output, self.hidden)
        return output

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # from pytorch documentation: (hidden state (h): (num_layers, mini_batch_size, hidden_dim),
        #                                cell state (c): (num_layers, mini_batch_size, hidden_dim)

        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device))


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        """
        decodes sampled input
        :param hidden_size: size of single hidden layer
        :param output_size: size of decoded output
        :param num_layers: number of hidden layers
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden = self.init_hidden()

    def forward(self, input):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        output = F.relu(input.view(1, 1, -1))
        output, self.hidden = self.lstm(output, self.hidden)
        output = self.softmax(self.out(output[0]))
        return output

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # from pytorch documentation: (hidden state (h): (num_layers, mini_batch_size, hidden_dim),
        #                                cell state (c): (num_layers, mini_batch_size, hidden_dim)

        return (torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
                torch.zeros(self.num_layers, 1, self.hidden_size, device=device))


class LSTM_VAE(nn.Module):
    def __init__(self, size, outputsize, num_rnn_layers=1):
        """
        :param size: window size of input
        :param outputsize: latent vector representation size * 2
        """
        super(LSTM_VAE, self).__init__()
        self.size = size
        self.outputsize = outputsize
        # self.encode_layer = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(size, 200),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(200, 200),
        #     nn.Dropout(),
        #     nn.Linear(200, outputsize),
        # )
        self.encode_layer = EncoderRNN(size, outputsize, num_rnn_layers).to(device)


        # self.decode_layer = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(self.outputsize // 2, 200),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(200, 200),
        #     nn.Dropout(),
        #     nn.Linear(200, self.size),
        # )
        self.decode_layer = DecoderRNN(outputsize // 2, size, num_rnn_layers).to(device)

        # hidden states initialization only once
        self.encode_layer.init_hidden()
        self.decode_layer.init_hidden()

    def forward(self, input):
        tmp = self.encode_layer(input)
        tmp = tmp.view(tmp.numel())
        mu, std = torch.split(tmp, self.outputsize // 2)
        sampled = self.sample(mu, std)
        output = self.decode_layer(sampled)
        return output, self.get_kl_loss(mu, std)

    def sample(self, mu, log_var):
        eps = torch.randn(self.outputsize // 2).cuda()
        return mu + torch.exp(log_var / 2) * eps

    def encode(self, input):
        tmp = self.encode_layer(input)
        mu, std = torch.split(tmp, self.outputsize // 2)
        return mu, std

    def decode(self, input):
        tmp = self.decode(input)
        return tmp

    def get_kl_loss(self, mu, std):
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(std) + mu ** 2 - 1. - mu, 0))
        return kl_loss


class RnnAnomalyDetector(object):
    def __init__(self, window_size, lr, diff_window_size, encoded_dim, num_rnn_layers=1):
        """
        :param window_size:
        :param lr:
        :param diff_window_size: dimension of diff window : used for mean and cov
        """
        self.window_size = window_size
        self.lr = lr
        self.diff_window_size = diff_window_size
        self.encoded_dim = encoded_dim
        # error window initialization
        self.diff_window = torch.zeros(diff_window_size, window_size).to(device)
        self.diff_counter = 0
        self.diff_mean = torch.zeros(window_size).to(device)
        self.diff_cov = torch.tensor(np.eye(window_size)).to(device)
        self.start = True

        self.model = LSTM_VAE(window_size, encoded_dim*2, num_rnn_layers)
        self.model.to(device)
        self.model.train()
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        self.optimizer.zero_grad()
        self.criterion =  nn.MSELoss()

    def reset_optimizer(self):
        """
        call this function if more passes through dataset are done
        """
        self.optimizer.zero_grad()

    def add_data_point(self, data_point):
        # TODO: dataset with all windows already in gpu memory --> or a part if the dataset is too big
        data = torch.tensor(data_point).to(device).float()

        #data = data.view(data.size()[0], -1)
        predicted, kl_loss = self.model(data)
        predicted = predicted.view(predicted.numel())
        error_score = self.criterion(predicted, data).detach()
        diff = predicted - data
        diff = diff.detach()

        loss = error_score + kl_loss
        loss.backward()
        self.optimizer.step()

        if self.diff_counter == self.diff_window_size:
            if self.start:
                self.start = False

            self.diff_counter = 0

            # update mean and cov
            self.diff_mean = self.diff_window.mean(0)
            self.diff_cov = torch.tensor(np.cov(self.diff_window.transpose(1, 0))).to(device)
        self.diff_window[self.diff_counter] = diff
        self.diff_counter += 1
        if self.start:
            score = 0
        else:
            score = - np.log(multivariate_normal.pdf(diff.cpu().detach().numpy(),
                                                 mean=self.diff_mean.cpu().detach().numpy(),
                                                 cov=self.diff_cov.cpu().detach().numpy())+1e-7)
        return score


if __name__=='__main__':
    window_size = 10
    lr = 0.001
    diff_window_size = 5
    encoded_dim = 2

    rnnDetector = RnnAnomalyDetector(window_size, lr, diff_window_size, encoded_dim)
    print(rnnDetector.add_data_point(np.linspace(0., 9., window_size)))

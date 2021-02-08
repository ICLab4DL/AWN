import argparse
import pickle
import numpy as np
import os

import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import linalg
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from pygsp import graphs, filters
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable
from sklearn.preprocessing import normalize
import time
import os
import util
import torch.optim as optim
import matplotlib.pyplot as plt
import pywt


class Sampling(nn.Module):
    def __init__(self, args, seq_len):
        super(Sampling, self).__init__()
        self.conv = nn.Conv1d(seq_len, args.att_out_channel, kernel_size=1)

    def forward(self, x):
        """
        :param x: (batch, N=1, channel, wavelet_seq)
        :return:  (batch, N=1, att_out_channel, wavelet_seq[-1])
        """
        x = x.squeeze()
        conv_out = self.conv(x)
        return conv_out[..., -1]


class WaveAttention(nn.Module):
    def __init__(self, args, seq_len):
        super(WaveAttention, self).__init__()
        self.args = args
        self.conv = nn.Conv1d(seq_len - 1, args.att_out_channel, kernel_size=1)
        self.Att_W = nn.Parameter(torch.FloatTensor(self.args.att_out_channel, args.rnn_output_channel))
        self.weight_init()

    def forward(self, x):
        """
        :param x: (batch, N, channel, wavelet_seq)
        :return:  (batch, N, att_out_channel + channel)
        """
        #batch, N, channel, seq_len = x.shape
        #x = x.reshape(batch * N, channel, seq_len).transpose(1, 2)
        
        batch, seq_len, N, in_channel = x.shape
        x = x.squeeze()
        if len(x.shape) < 3:
            # this is health condition:
            x = x.unsqueeze(dim=1)
        #print("x shape", x.shape)

        
        att_x = x[:, :-1, :]
        #print("att_x shape", att_x.shape) #[256,44,3]
        h_t = x[:, -1, :].unsqueeze(dim=2)
        #print("h_t shape", h_t.shape) #[256,3,1]
        conv_out = self.conv(att_x).transpose(1, 2)
        #print("conv_out shape", conv_out.shape) #[256,3,20]
        #print("self.Att_W shape", self.Att_W.shape) #[20,64]
        # batch x N, out_channel, seq_len
        att1 = torch.einsum('bos, si -> boi', conv_out, self.Att_W)
        #print("att1 shape", att1.shape) #[256,3,64]
        att2 = torch.einsum('boi, bij -> boj', att1.transpose(1, 2), h_t).squeeze()
        #print("att2 shape", att2.shape) #[256,64]

        #a = torch.sigmoid(att2).unsqueeze(dim=1)
        #print("a shape", a.shape) #[256,1,64]
        #v = torch.einsum('biv, bvk -> bik', a, conv_out.transpose(1, 2)).squeeze()
        #print("v shape", v.shape)
        #out = torch.cat([v, h_t.squeeze()], dim=1).reshape(batch, N, -1)
        #print("att out shape", out.shape)
        out = torch.sigmoid(att2)
        #print("out shape", out.shape)
        
        return out

    def weight_init(self):
        nn.init.normal_(self.Att_W, mean=0.0, std=0.001)
        nn.init.xavier_uniform_(self.conv.weight)


class FFNN(nn.Module):
    def __init__(self, args):
        super(FFNN, self).__init__()
        self.args = args
        self.factor_num = args.factor_num
        print('FFNN :',args.rnn_output_channel)
        self.encoder0 = EncoderLSTM(args, args.feature_len, args.rnn_hidden_len, args.rnn_output_channel, args.rnn_layer_num)
        self.encoder1 = EncoderLSTM(args, args.feature_len, args.rnn_hidden_len, args.rnn_output_channel, args.rnn_layer_num)
        self.encoder2 = EncoderLSTM(args, args.feature_len, args.rnn_hidden_len, args.rnn_output_channel, args.rnn_layer_num)
        self.encoder3 = EncoderLSTM(args, args.feature_len, args.rnn_hidden_len, args.rnn_output_channel, args.rnn_layer_num)
        self.encoder4 = EncoderLSTM(args, args.feature_len, args.rnn_hidden_len, args.rnn_output_channel, args.rnn_layer_num)
        
        self.seq_len0 = args.seq_len0
        self.seq_len1 = args.seq_len1
        self.seq_len2 = args.seq_len2
        self.seq_len3 = args.seq_len3
        self.seq_len4 = args.seq_len4

        self.simple0 = SimpleModel(args, args.seq_len0, input_feature=args.feature_len, output_feature=32)
        self.simple1 = SimpleModel(args, args.seq_len1, input_feature=args.feature_len, output_feature=32)
        self.simple2 = SimpleModel(args, args.seq_len2, input_feature=args.feature_len, output_feature=32)
        self.simple3 = SimpleModel(args, args.seq_len3, input_feature=args.feature_len, output_feature=32)
        self.simple4 = SimpleModel(args, args.seq_len4, input_feature=args.feature_len, output_feature=32)
        
        self.waveconvatt0 = WaveConvAtt(args, args.seq_len0, input_feature=args.feature_len, output_feature=32)
        self.waveconvatt1 = WaveConvAtt(args, args.seq_len1, input_feature=args.feature_len, output_feature=32)
        self.waveconvatt2 = WaveConvAtt(args, args.seq_len2, input_feature=args.feature_len, output_feature=32)
        self.waveconvatt3 = WaveConvAtt(args, args.seq_len3, input_feature=args.feature_len, output_feature=32)
        self.waveconvatt4 = WaveConvAtt(args, args.seq_len4, input_feature=args.feature_len, output_feature=32)

        self.attention_x0 = WaveAttention(args, seq_len=args.seq_len0)
        self.attention_xA3 = WaveAttention(args, seq_len=args.seq_len1)
        self.attention_xD3 = WaveAttention(args, seq_len=args.seq_len2)
        self.attention_xD2 = WaveAttention(args, seq_len=args.seq_len3)
        self.attention_xD1 = WaveAttention(args, seq_len=args.seq_len4)
        
        self.fc_x0_1 = nn.Linear( 5* args.rnn_hidden_len, 256) 
        self.fc_x0_2 = nn.Linear(256, args.rnn_hidden_len)
        self.fc_a3_1 = nn.Linear( 5* args.rnn_hidden_len, 256)
        self.fc_a3_2 = nn.Linear(256, args.rnn_hidden_len)
        self.fc_d3_1 = nn.Linear( 5* args.rnn_hidden_len, 256)
        self.fc_d3_2 = nn.Linear(256, args.rnn_hidden_len)
        self.fc_d2_1 = nn.Linear( 5* args.rnn_hidden_len, 256)
        self.fc_d2_2 = nn.Linear(256, args.rnn_hidden_len)
        self.fc_d1_1 = nn.Linear( 5* args.rnn_hidden_len, 256)
        self.fc_d1_2 = nn.Linear(256, args.rnn_hidden_len)
        
        
        self.fc_cat_1 = nn.Linear( 4* args.rnn_hidden_len, 256)
        self.fc_cat_2 = nn.Linear(256, 4 * args.rnn_hidden_len)
        
        self.fc1 = nn.Linear(4*args.rnn_hidden_len, args.predict_label_num)
       

        self.simple_last = SimpleModel(args, 1, input_feature=4 * 32, output_feature=args.predict_label_num)
        self.simple_last_x0 = SimpleModel(args, args.seq_len0, input_feature=args.feature_len, output_feature=args.predict_label_num)
        self.weight_init()

    def forward(self, x):
        """
        :param x: (batch, seq_len, N, in_channel)
        """
        x0, xA3, xD3, xD2, xD1 = x
        
        batch, seq_len_xA3, N, in_channel = xA3.shape
        # print("wavelet_trans input shape:", x.shape)
        xA3_lstm = self.encoder1(xA3)
        xD3_lstm = self.encoder2(xD3)        
        xD2_lstm = self.encoder3(xD2)        
        xD1_lstm = self.encoder4(xD1)
        x0_lstm = self.encoder0(x0)
        
       
        xA3_out = xA3
        xD3_out = xD3
        xD2_out = xD2
        xD1_out = xD1
        x0_out = x0
       
        
        xA3_conv = self.waveconvatt1(xA3_out)
        xD3_conv = self.waveconvatt2(xD3_out)
        xD2_conv = self.waveconvatt3(xD2_out)
        xD1_conv = self.waveconvatt4(xD1_out)
        x0_conv = self.waveconvatt0(x0_out)
       
        catted= torch.cat([xA3_conv, xD3_conv, xD2_conv, xD1_conv], dim=1) 

        a3_att = torch.softmax(F.relu(self.fc_a3_2(F.relu(self.fc_a3_1(torch.cat([xA3_lstm, catted], dim=1))))),dim=1) 
        a3_out = a3_att * x0_conv 
    
        d3_att = torch.softmax(F.relu(self.fc_d3_2(F.relu(self.fc_d3_1(torch.cat([xD3_lstm, catted], dim=1))))),dim=1)
        d3_out = d3_att * x0_conv
        
        d2_att = torch.softmax(F.relu(self.fc_d2_2(F.relu(self.fc_d2_1(torch.cat([xD2_lstm, catted], dim=1))))),dim=1)
        d2_out = d2_att * x0_conv

        d1_att = torch.softmax(F.relu(self.fc_d1_2(F.relu(self.fc_d1_1(torch.cat([xD1_lstm, catted], dim=1))))),dim=1)
        d1_out = d1_att * x0_conv

        out = torch.cat([a3_out, d3_out, d2_out, d1_out], dim=1)
        out = F.relu(self.fc1(out))
    

        return out        
        
        


class EncoderLSTM(nn.Module):
    def __init__(self, args, feature_len, hidden_len, rnn_output_channel, num_layers=1, bidirectional=False):
        super(EncoderLSTM, self).__init__()
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.rnn_output_channel =  rnn_output_channel
        # RNN层
        self.rnn = nn.LSTM(
            input_size=feature_len,  
            hidden_size=hidden_len,  
            num_layers=num_layers, 
            batch_first=True,  
            bidirectional=bidirectional
        )

        if bidirectional:
            self.conv = nn.Conv1d(2 * hidden_len, hidden_len, kernel_size=1)
        else:
            self.conv = nn.Conv1d(hidden_len, hidden_len, kernel_size=1)
        #self.lbn = nn.LayerNorm([hidden_len, args.origin_seq_len])
        self.bn1 = nn.BatchNorm1d(hidden_len)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        """
        x = (batch, seq, N, channel)
        to x = (batch x N, sequence, channel)
        :return:输出out(batch, N, rnn_output_channel, sequence)
        """
        batch, seq, N, channel = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(batch * N, seq, channel)
        out, _ = self.rnn(x)
        out = out[:,-1,:].unsqueeze(dim=1)
        # out  = batch*N, seq, hidden_num
        out = out.transpose(1, 2)
        out = self.bn1(out)
        out = self.dropout(out)
        out = out.reshape(batch, self.hidden_len)
        return out


class SimpleModel(nn.Module):
    def __init__(self, args, seq_len, input_feature, output_feature):
        super(SimpleModel, self).__init__()
        self.args = args
        self.factor_num = args.factor_num
        self.conv1 = nn.Conv1d(input_feature, 16, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        # self.encoder = EncoderLSTM(args.feature_len, args.rnn_hidden_len, args.rnn_output_channel, args.rnn_layer_num)
        self.encoder = EncoderLSTM(args, 4, args.rnn_hidden_len, args.rnn_output_channel,
                                   args.rnn_layer_num, bidirectional=args.bidirectional)
        # health:
        self.fc1 = nn.Linear(seq_len * args.rnn_output_channel, args.predict_label_num)
        self.fc2 = nn.Linear(seq_len * 32, output_feature)
        self.dropout3 = nn.Dropout(0.2)
        #         self.fc1 = nn.Linear(4 * (args.rnn_output_channel + args.att_out_channel), 5)
        # add a sigmoid to generate probability
        # self.fc2 = nn.Linear()
        self.weight_init()

    def forward(self, x):
        """
        :param x: (batch, sequence, N, channel)
         returns : (batch, channel)
        """
        # print("SimpleModel input: ", x.shape)
        batch, seq_len, N, in_channel = x.shape
        x = x.squeeze()
        if len(x.shape) < 3:
            # this is health condition:
            x = x.unsqueeze(dim=1)

        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout2(x)

   
        x = x.squeeze()
        x = x.reshape(batch, -1)

        out = self.fc2(x)
        out = F.relu(out)
        out = self.dropout3(out)
        #         out = F.relu(self.fc2(out))
        return out

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)


class Trainer:
    def __init__(self, args, model, optimizer, criterion=nn.BCELoss()):
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.clip = args.clip
        self.lr_decay_rate = args.lr_decay_rate
        self.epochs = args.epochs
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epochs: self.lr_decay_rate ** epochs)

    def train(self, input_data, target, need_print=False):
        self.model.train()
        self.optimizer.zero_grad()

        # train
        output = self.model(input_data)
        output = output.squeeze()
      
        loss, acc = util.calc_metrics_multi_class(output, target, criterion=self.criterion, cuda=self.args.cuda, need_print=need_print)
        
        regularization_loss = 0
#         for param in self.model.parameters():
#             regularization_loss += torch.sum(abs(param))
#         loss = loss + 0.001 * regularization_loss
        #loss, acc = util.calc_metrics_multi_class(output, target, criterion=self.criterion, cuda=self.args.cuda, need_print=need_print)
        loss.backward(retain_graph=True)
        # loss.backward()
        self.optimizer.step()
        return loss.item(), acc.item()

    def eval(self, input_data, target, need_print=False):
        self.model.eval()
        output = self.model(input_data)  # [batch_size,seq_length,num_nodes]
        output = output.squeeze()
        mae, acc = util.calc_metrics_multi_class(output, target, criterion=self.criterion, cuda=self.args.cuda, need_print=need_print)
        return mae.item(), acc.item()

    def predict(self, input_data):
        self.model.eval()
        return self.model(input_data)

    
    
class WaveConvAtt(nn.Module):
    def __init__(self, args, seq_len, input_feature, output_feature):
        super(WaveConvAtt, self).__init__()
        self.args = args
        self.factor_num = args.factor_num

        self.conv1 = nn.Conv1d(input_feature, 16, kernel_size=1) 
       
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=1)

        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(input_feature, 32, kernel_size=seq_len)
        self.bn3 = nn.BatchNorm1d(32)
        
        


        self.encoder = EncoderLSTM(args, 4, args.rnn_hidden_len, args.rnn_output_channel,
                                   args.rnn_layer_num, bidirectional=args.bidirectional)
        
        

        
        self.fc1 = nn.Linear(seq_len * 32, 256)  #目前最好的模型這裡是32，128
        self.fc2 = nn.Linear(256, args.rnn_hidden_len)
        self.fc3 = nn.Linear(seq_len * 32, args.rnn_hidden_len)
        self.fc4 = nn.Linear(32, 256)
        self.fc5 = nn.Linear(256, args.rnn_hidden_len)
        
        self.dropout3 = nn.Dropout(0.2)
        #         self.fc1 = nn.Linear(4 * (args.rnn_output_channel + args.att_out_channel), 5)
        # add a sigmoid to generate probability
        # self.fc2 = nn.Linear()
        self.weight_init()

    def forward(self, x):
        """
        :param x: (batch, sequence, N, channel)
         returns : (batch, channel)
        """
        # print("SimpleModel input: ", x.shape)
        batch, seq_len, N, in_channel = x.shape
        x = x.squeeze()
        if len(x.shape) < 3:
            # this is health condition:
            x = x.unsqueeze(dim=1)
        
        xx = x.transpose(1, 2)

        xx = self.conv1(xx)
        xx = F.relu(self.bn1(xx))
        xx = self.dropout1(xx)

        xx = self.conv2(xx)
        xx = F.relu(self.bn2(xx))
        xx = self.dropout2(xx)
        

        xx = xx.reshape(batch, -1)


        out = F.relu(self.fc2(F.relu(self.fc1(xx))))
        
    
        
        return out

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
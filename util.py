import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

data_len = 470 #260


def get_common_args():
    parser = argparse.ArgumentParser()
    # learning params
    parser.add_argument('--dev', action='store_true', help='dev')
    parser.add_argument('--dev_size', type=int, default=1000, help='dev_sample_size')
    parser.add_argument('--best_model_save_path', type=str, default='.best_model', help='best_model')
    parser.add_argument('--pre_model_path', type=str, default='./pre_model/best_model', help='pre_model_path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=40, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')    
    parser.add_argument('--lr_decay_rate', type=float, default=0.985, help='lr_decay_rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
    parser.add_argument('--clip', type=int, default=3, help='clip')
    parser.add_argument('--seq_length', type=int, default=30, help='seq_length')
    parser.add_argument('--predict_len', type=int, default=1, help='predict_len')
    parser.add_argument('--scheduler', action='store_true', help='scheduler')
    parser.add_argument('--mo', type=float, default=0.1, help='momentum')

    # running params
    parser.add_argument('--cuda', action='store_true', help='cuda')
    parser.add_argument('--transpose', action='store_true', help='transpose sequence and feature?')
    parser.add_argument('--data_path', type=str, default='./260all_crossval/', help='data path')
    parser.add_argument('--data_type', type=str, default='health', help='data type')

    # model params
    parser.add_argument('--pretrain', action='store_true', help='pretrain')
    parser.add_argument('--origin_seq_len', type=int, default=440, help='origin_seq_len') ###data len
    parser.add_argument('--bidirectional', action='store_true', help='bidirectional')
    parser.add_argument('--predict_label_num', type=int, default=5, help='predict_label_num') ## number of labels

    # health:
    parser.add_argument('--feature_len', type=int, default=3, help='input feature_len') #2 IC +1 LF
    parser.add_argument('--factor_num', type=int, default=1, help='factor_num')


    parser.add_argument('--rnn_hidden_len', type=int, default=32, help='rnn hidden_len') 
    parser.add_argument('--rnn_layer_num', type=int, default=2, help='rnn_layer_num')
    parser.add_argument('--rnn_output_channel', type=int, default=5, help='rnn_output_channel') ## number of labels
    parser.add_argument('--att_in_channel', type=int, default=37, help='att_in_channel')
    parser.add_argument('--att_out_channel', type=int, default=20, help='att_out_channel')

    # health:

    CD1_len = int(np.floor((data_len + 16 - 1) / 2))
    CD2_len = int(np.floor((CD1_len + 16 - 1) / 2))
    CD3_len = int(np.floor((CD2_len + 16 - 1) / 2))
    CA3_len = CD3_len
    C0_len = data_len
    
    parser.add_argument('--seq_len0', type=int, default=C0_len, help='seq_len0')
    parser.add_argument('--seq_len1', type=int, default=CA3_len, help='seq_len1')
    parser.add_argument('--seq_len2', type=int, default=CD3_len, help='seq_len2')
    parser.add_argument('--seq_len3', type=int, default=CD2_len, help='seq_len3')
    parser.add_argument('--seq_len4', type=int, default=CD1_len, help='seq_len4')

    return parser


class StandardScaler():

    def __init__(self, mean, std, fill_zeroes=True):
        self.mean = mean
        self.std = std
        self.fill_zeroes = fill_zeroes

    def transform(self, data):
        if self.fill_zeroes:
            mask = (data == 0)
            data[mask] = self.mean
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class HeartDataLoader(object):
    def __init__(self, x0, xA3, xD3, xD2, xD1, ys, batch_size, cuda=False, transpose=False, pad_with_last_sample=False):
        """
        :param xs: ()
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        print('in x0', x0.shape)
        print('in xA3', xA3.shape)
        print('in xD3', xD3.shape)
        print('in xD2', xD2.shape)
        print('in xD1', xD1.shape)
        self.batch_size = batch_size
        self.current_ind = 0

        self.size = len(xA3)
        self.num_batch = int(self.size // self.batch_size)
        x0 = torch.Tensor(x0)
        xA3 = torch.Tensor(xA3)
        xD3 = torch.Tensor(xD3)
        xD2 = torch.Tensor(xD2)
        xD1 = torch.Tensor(xD1)
        ys = torch.Tensor(ys)
        
        if cuda:
            x0 = x0.cuda()
            xA3 = xA3.cuda()
            xD3 = xD3.cuda()
            xD2 = xD2.cuda()
            xD1 = xD1.cuda()
            ys = ys.cuda()
        if transpose:
            x0 = x0.transpose(1, 3)
            xA3 = xA3.transpose(1, 3)
            xD3 = xD3.transpose(1, 3)
            xD2 = xD2.transpose(1, 3)
            xD1 = xD1.transpose(1, 3)
            # ys = ys.transpose(1, 2)
        self.x0 = x0
        self.xA3 = xA3
        self.xD3 = xD3
        self.xD2 = xD2
        self.xD1 = xD1

        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        x0 = self.x0[permutation]
        xA3 = self.xA3[permutation]
        xD3 = self.xD3[permutation]
        xD2 = self.xD2[permutation]
        xD1 = self.xD1[permutation]
        ys = self.ys[permutation]
        self.x0 = x0
        self.xA3 = xA3
        self.xD3 = xD3
        self.xD2 = xD2
        self.xD1 = xD1
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            start_ind = 0
            end_ind = 0
            while self.current_ind < self.num_batch and start_ind <= end_ind and start_ind <= self.size:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x0_i = self.x0[start_ind: end_ind, ...]
                xA3_i = self.xA3[start_ind: end_ind, ...]
                xD3_i = self.xD3[start_ind: end_ind, ...]
                xD2_i = self.xD2[start_ind: end_ind, ...]
                xD1_i = self.xD1[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x0_i, xA3_i, xD3_i, xD2_i, xD1_i, y_i
                self.current_ind += 1

        return _wrapper()


def norm(tensor_data, dim=0):
    mu = tensor_data.mean(axis=dim, keepdim=True)
    std = tensor_data.std(axis=dim, keepdim=True)
    return (tensor_data - mu) / (std + 0.00005)


def calc_metrics(preds, labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    # handle all zeros.
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = (preds - labels) ** 2
    mae = torch.abs(preds - labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
    rmse = torch.sqrt(mse)

    # accuracy
    label_bi = labels.squeeze() > 0
    preds_bi = preds > 0
    total = torch.sum(label_bi == preds_bi).float()
    acc = total / preds.numel()
    return mae, mape, rmse, acc


class MetricCollector:
    """
    MetricCollector
    ------------------
    This class is used for collecting metrics you need in result analysis.
    You can put a dict like data structure in this class.
    ------------------
    """

    def __init__(self):
        self.metrics = dict()

    def create_metrics(self, metric_label, key):
        self.metrics[metric_label] = dict()
        self.metrics[metric_label][key] = []

    def put(self, metric_label, key, value):
        if not self.metrics.__contains__(metric_label):
            self.create_metrics(metric_label, key)
        if not self.metrics[metric_label].__contains__(key):
            self.metrics[metric_label][key] = []
        self.metrics[metric_label][key].append(value)

    def get(self, metric_label, key):
        return self.metrics[metric_label][key]


def calc_metrics_binary_val(preds, labels, criterion):
    """
        return BCE, ACC (List)
    """
    # method 1: cross entropy:
    labels = labels.squeeze()
    labels = labels > 0
    labels = labels.long().unsqueeze(dim=2)
    b, N, _ = labels.shape
    one_hot = torch.zeros(b, N, 2).scatter_(2, labels, torch.ones(b, N, 1))
    one_hot = one_hot.float()
    bce = criterion(preds, one_hot)

    # calculate acc:
    preds_b = preds.argmax(dim=2)
    one_hot_b = one_hot.argmax(dim=2)
    total = torch.sum(one_hot_b == preds_b, dim=0).float()
    acc = total / b
    return bce, acc


def calc_metrics_multi_class(preds, labels, criterion, metrics_collector=None, cuda=False, need_print=True):
    labels = labels.long()
   
    
    batch, _ = preds.shape
    # print("label shape:", labels.shape)
    # print("preds shape:", preds.shape)

    # to one hot:
    # labels = labels.unsqueeze(dim=1)
    # b = labels.shape[0]
    # zeros = torch.zeros(b, 5)
    # if cuda:
    #     zeros = zeros.cuda()
    # one_hot = zeros.scatter_(1, labels, torch.ones(b, 1)).long()
    # print("one_hot: ", one_hot.shape)
    
    
    loss = criterion(preds, labels)
    
  
    preds_b = preds.argmax(dim=1) 
    correct = torch.sum(labels == preds_b).float()
    if need_print:
        print("correct: ", correct)
        print("batch: ", batch)
    acc = correct / batch

    if metrics_collector is not None:
        metrics_collector.put("test", "acc", acc)
    return loss, acc


def calc_metrics_binary(preds, labels, criterion, cuda=False):
    # method 1: cross entropy:
    labels = labels.squeeze()
    labels = labels > 0
    labels = labels.long().unsqueeze(dim=2)
    b, N, _ = labels.shape
    zeros = torch.zeros(b, N, 2)
    if cuda:
        zeros = zeros.cuda()
    one_hot = zeros.scatter_(2, labels, torch.ones(b, N, 1))
    one_hot = one_hot.float()
    bce = criterion(preds, one_hot)

    # calculate acc:
    preds_b = preds.argmax(dim=2)
    one_hot_b = one_hot.argmax(dim=2)
    total = torch.sum(one_hot_b == preds_b).float()
    acc = total / (b * N)
    return bce, acc


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_tstep_metrics(model, test_loader, scaler, realy, seq_length) -> pd.DataFrame:
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).cuda().transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    for i in range(seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        pred = torch.clamp(pred, min=0., max=70.)
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    return test_met_df, yhat

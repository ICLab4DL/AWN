import sys

# sys.setdefaultencoding('utf8')

from torch import optim

from util import *
from model import *
import matplotlib.pyplot as plt
import time
import os
from model.FFNN import *
from fastprogress import progress_bar


import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(threshold=np.inf)


def get_scalers(data):
    scalers = []
    shape = data.shape
    for i in range(shape[-1]):
        scalers.append(StandardScaler(mean=data[..., i].mean(), std=data[..., i].std()))
    return scalers


def load_dataset(args):
    datasets = {}
    dir = args.data_path
    print("Loading from path:", dir)
    batch_size = args.batch_size

    # we need to filter first and last value in data.

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dir, category + '.npz'), allow_pickle=True)
        if args.dev:
            datasets['x_' + category] = cat_data['x'][:args.dev_size]
        else:
            datasets['x_' + category] = np.array(cat_data['x'], dtype=np.float)
            print(category + ' x size: ', datasets['x_' + category].shape)

        if args.dev:
            datasets['y_' + category] = cat_data['y'][:args.dev_size, ..., 0]
        else:
            datasets['y_' + category] = np.array(cat_data['y'], dtype=np.float)
            print(category + ' y size: ', datasets['y_' + category].shape)

        if args.data_type == "health":
            datasets['x_' + category] = np.expand_dims(datasets['x_' + category], axis=2)
            if args.feature_len < 2:
                datasets['x_' + category] = np.expand_dims(datasets['x_' + category], axis=3)
                print('health, expanding dimension,---> ' + category + ' x size: ', datasets['x_' + category].shape)

        # for d in datasets["x_train"]:
        #     plt.plot(d.squeeze()[..., 0])
        #     plt.savefig('before_val_figure_1')
        #     plt.plot(d.squeeze()[..., 1])
        #     plt.savefig('before_val_figure_2')
        #     break
    # normalization of first feature: speed

    scalers = get_scalers(datasets['x_train'])

    # construct dataloader
    for category in ['train', 'val', 'test']:
        # norm each feature
        data = datasets['x_' + category]
        # for i in range(shape[-1]):
        # data[..., i] = scalers[i].transform(data[..., i])
        # construct data
        print("start extracting factor for: ", category)
        print("data shape", data.shape)
        x0, xA3, xD3, xD2, xD1 = wavelet_trans(args, data)
        #xA3, xD3, xD2, xD1 = wavelet_trans(args, data)
        
        print("x0 shape", x0.shape)
        print("xA3 shape", xA3.shape)
        print("xD3 shape", xD3.shape)
        print("xD2 shape", xD2.shape)
        print("xD1 shape", xD1.shape)
        # b, seq, N, channel
#       extract_factor      :param x: batch, N, channel, seq
#     :return: batch, N, channel + fac_num, seq

        x0 = x0.transpose(1, 2).transpose(2,3)
        xA3 = xA3.transpose(1, 2).transpose(2,3)
        xD3 = xD3.transpose(1, 2).transpose(2,3)
        xD2 = xD2.transpose(1, 2).transpose(2,3)
        xD1 = xD1.transpose(1, 2).transpose(2,3)
        
        x0 = extract_factor(args, x0)
        xA3 = extract_factor(args, xA3)
        xD3 = extract_factor(args, xD3)
        xD2 = extract_factor(args, xD2)
        xD1 = extract_factor(args, xD1)
        
        print("x0 shape1:", x0.shape)
        print("xA3 shape1:", xA3.shape)
        print("xD3 shape1:", xD3.shape)
        print("xD2 shape1:", xD2.shape)
        print("xD1 shape1:", xD1.shape)
        
        # b, s, N, c
        x0 = x0.transpose(1, 3).transpose(2,3)
        xA3 = xA3.transpose(1, 3).transpose(2,3)
        xD3 = xD3.transpose(1, 3).transpose(2,3)
        xD2 = xD2.transpose(1, 3).transpose(2,3)
        xD1 = xD1.transpose(1, 3).transpose(2,3)
        
        # (batch, sequence, N, channel)
        print("data shape after wavelet trans: ", xA3.shape)

        datasets[category + '_loader'] = HeartDataLoader(x0, xA3, xD3, xD2, xD1, datasets['y_' + category],
                                                         batch_size, args.cuda, transpose=args.transpose)
        

    # check data:
    print_info(datasets["train_loader"], "train data")
    print_info(datasets["test_loader"], "test data")
    print_info(datasets["val_loader"], "val data")

    # plot:
    #     for _, (input_data, target) in enumerate(val_data.get_iterator()):
    #         print(input_data.shape)
    #         plt.plot(input_data.squeeze()[0, ..., 0].numpy())
    #         plt.savefig('val_figure_1')
    #         plt.plot(input_data.squeeze()[0, ..., 1].numpy())
    #         plt.savefig('val_figure_2')
    #         break

    print('finish load dataset!')
    return datasets, scalers


def print_info(data_set, name):
    ts = dict()
    for _, (_, _, _, _, _, target) in enumerate(data_set.get_iterator()):
        for t in target:
            t = t.item()
            if ts.__contains__(t):
                ts[t] += 1
            else:
                ts[t] = 1
    print(f"The {name} label and it's corresponding number: ")
    for k, v in ts.items():
        print("label:", k, "number:", v)


#def training(args, datasets, scalers, metrics_collector):
def training(args, datasets, scalers, metrics_collector, K):
    """
    training the model
    Parameters
    ----------
    args:               all the parameters
    datasets:           all the data
    scalers:            the scalers of data
    metrics_collector:  refer to util.MetricCollector
    Returns
    -------

    """
    #     model = SimpleModel(args, args.origin_seq_len)
    model = FFNN(args)
    if args.pretrain:
        print('pretrainok')
        model.load_state_dict(torch.load(args.pre_model_path))

    print('args_cuda:', args.cuda)
    if args.cuda:
        print('rnn_train RNNBlock to cuda!')
        model.cuda()
    else:
        print('rnn_train RNNBlock to cpu!')

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # only for health:
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    


    trainer = Trainer(args, model, optimizer, criterion=criterion)

    best_model = dict()
    best_val_mae = 1000
    best_unchanged_threshold = 100  # accumulated epochs of best val_mae unchanged
    best_count = 0
    best_index = -1
    train_val_metrics = []
    start_time = time.time()

    for e in range(args.epochs):
        print('Starting epoch: ', e)
        datasets['train_loader'].shuffle()
        
        train_mae, train_acc = [], []
        need_print = False
        for i, (x0, xA3, xD3, xD2, xD1, target) in enumerate(datasets['train_loader'].get_iterator()):
   
            if args.cuda:
                x0, xA3, xD3, xD2, xD1 = x0.cuda(), xA3.cuda(), xD3.cuda(), xD2.cuda(), xD1.cuda()

                target = target.cuda()
            x0, xA3, xD3, xD2, xD1 = Variable(x0), Variable(xA3), Variable(xD3), Variable(xD2), Variable(xD1)
       
            target = Variable(target)

            
            if i % 50 ==0:
                #print("-------train------------")
                #need_print= True
                need_print= False
            mae, acc = trainer.train((x0, xA3, xD3, xD2, xD1), target, need_print)
            need_print = False
            # training metrics
            train_mae.append(mae)
            train_acc.append(acc)

        # validation metrics
        # TODO: pick best model with best validation evaluation.
        datasets['val_loader'].shuffle()

        val_mae, val_acc = [], []
        need_print = False

        for j, (x0, xA3, xD3, xD2, xD1, target) in enumerate(datasets['val_loader'].get_iterator()):
     
            if args.cuda:
                # input_data = input_data.cuda()
                x0, xA3, xD3, xD2, xD1 = x0.cuda(), xA3.cuda(), xD3.cuda(), xD2.cuda(), xD1.cuda()

                target = target.cuda()
            # input_data, target = Variable(input_data), Variable(target)
            x0, xA3, xD3, xD2, xD1 = Variable(x0), Variable(xA3), Variable(xD3), Variable(xD2), Variable(xD1)
            
            target = Variable(target)
            # input_data, target = Variable(input_data), Variable(target)
            
            if j % 50 ==0:
                print("-------val------------")
                #need_print= True
                need_print= False
            mae, acc = trainer.eval((x0, xA3, xD3, xD2, xD1), target, need_print)
         
            need_print = False
            # add metrics
            val_mae.append(mae)
          
            val_acc.append(acc)


        m = dict(train_mae=np.mean(train_mae), train_acc=np.mean(train_acc),
                 valid_mae=np.mean(val_mae), valid_acc=np.mean(val_acc))

        m = pd.Series(m)
        #         valid_acc_str = [str(a) for a in m['valid_acc']]
        #         print(",".join(valid_acc_str))
        print(m)
        
        train_val_metrics.append(m)
        # once got best validation model ( 20 epochs unchanged), then we break.
        if m['valid_mae'] < best_val_mae:
            best_val_mae = m['valid_mae']
            best_count = 0
            best_model = trainer.model.state_dict()
            best_index = e
        else:
            best_count += 1
        if best_count > best_unchanged_threshold:
            print('Got best')
            break
    #         trainer.scheduler.step()

    # test metrics
    torch.save(best_model, args.best_model_save_path)
    trainer.model.load_state_dict(torch.load(args.best_model_save_path))
    print('best_epoch:', best_index)
    
    datasets['test_loader'].shuffle()
    test_mae, test_acc = [], []
    need_print = False
    # for i, (input_data, target) in enumerate(datasets['test_loader'].get_iterator()):
    
    for i, (x0, xA3, xD3, xD2, xD1, target) in enumerate(datasets['test_loader'].get_iterator()):
        if args.cuda:
            # input_data = input_data.cuda()
            x0, xA3, xD3, xD2, xD1 = x0.cuda(), xA3.cuda(), xD3.cuda(), xD2.cuda(), xD1.cuda()
         
            target = target.cuda()
        x0, xA3, xD3, xD2, xD1 = Variable(x0), Variable(xA3), Variable(xD3), Variable(xD2), Variable(xD1)
    
        target = Variable(target)
        # input_data, target = Variable(input_data), Variable(target)
        if j % 10 ==0:
            #print("-------test------------")
            need_print= False
            #need_print= True
        mae, acc = trainer.eval((x0, xA3, xD3, xD2, xD1), target, need_print)
       
        need_print = False
        # add metrics
        test_mae.append(mae)
        test_acc.append(acc)
    m = dict(test_mae=np.mean(test_mae), test_acc=np.mean(test_acc))
    m = pd.Series(m)
    print("test:")
    print(m)
    plot(train_val_metrics)
    plot_acc(train_val_metrics)

    # TODO: add test metrics into metrics_collector, Precision,
    print("test_acc: ", m["test_acc"])

    metrics_collector.put("test", "acc", m["test_acc"])
    metrics_collector.put("test", "mae", m["test_mae"])

    print("one metrics_collector: ", metrics_collector.get("test", "acc"))
    print("one metrics_collector mean: ", np.mean(metrics_collector.get("test", "acc"), axis=0))

    # ------------------------ end -----------------------------

    ts = dict()
    td = dict()
    Target, Output, Preds = [],[],[]
    for j, (x0, xA3, xD3, xD2, xD1, target)  in enumerate(datasets['test_loader'].get_iterator()):
  
    # for j, (input_data, target) in enumerate(datasets['test_loader'].get_iterator()):
        if args.cuda:
            # input_data = input_data.cuda()
            x0, xA3, xD3, xD2, xD1 = x0.cuda(), xA3.cuda(), xD3.cuda(), xD2.cuda(), xD1.cuda()
            
            target = target.cuda()
        x0, xA3, xD3, xD2, xD1 = Variable(x0), Variable(xA3), Variable(xD3), Variable(xD2), Variable(xD1)
      
        target = Variable(target)
        Target.append(target.cpu().detach().numpy())
        
        # input_data, target = Variable(input_data), Variable(target)
        output = trainer.predict((x0, xA3, xD3, xD2, xD1)).squeeze()
      
        Output.append(output.cpu().detach().numpy())
        
        preds_b = output.argmax(dim=1)
        Preds.append(preds_b.cpu().detach().numpy())
        
        
        for i in range(preds_b.shape[0]):
            t = preds_b[i].item()
            if ts.__contains__(t):
                ts[t] += 1
            else:
                ts[t] = 1

        for i in range(target.shape[0]):
            t = target[i].item()
            if td.__contains__(t):
                td[t] += 1
            else:
                td[t] = 1

    for k, v in ts.items():
        print("test label:", k, "number:", v)

    np.save("train_val_metrics_%s.npy" %K, train_val_metrics)
    np.save("target_%s.npy" %K,Target)
    np.save("output_%s.npy" %K,Output)
    np.save("preds_%s.npy" %K,Preds)

    print('finish rnn_train!, time cost:', time.time() - start_time)


def plot(train_val_metrics):
    epochs = len(train_val_metrics)
    x = range(epochs)
    train_mae = [m['train_mae'] for m in train_val_metrics]
    val_mae = [m['valid_mae'] for m in train_val_metrics]
    plt.figure(figsize=(8, 6))
    plt.plot(x, train_mae, '', label='train_mae',linewidth=2.5)
    plt.plot(x, val_mae, '', label='val_mae',linewidth=2.5)
    plt.title('MAE', fontsize=18)
    plt.legend(loc='lower right', fontsize=14) 
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('mae', fontsize=16)
    plt.grid()
    plt.savefig('mae')

    
def plot_acc(train_val_metrics):
    epochs = len(train_val_metrics)
    x = range(epochs)
    train_acc = [m['train_acc'] for m in train_val_metrics]
    val_acc = [m['valid_acc'] for m in train_val_metrics]
    plt.figure(figsize=(8, 6))
    plt.plot(x, train_acc, '', label='train_acc',linewidth=2.5)
    plt.plot(x, val_acc, '', label='val_acc',linewidth=2.5)
    plt.title('ACC', fontsize=18)
    plt.legend(loc='lower right', fontsize=14) 
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('acc', fontsize=16)
    plt.grid()
    plt.savefig('acc')
    
    
    

def cross_validation_analysis(metrics):
    """
    calculate F1, A, R, etc., using K-fold metrics.
    Parameters
    ----------
    metrics:    MetricsCollector

    Returns:   Analysis
    -------
    """
    accs = []
    for m in metrics:
        accs.append(m.get("test", "acc"))
    print("cross avg acc: ", np.mean(np.concatenate(accs, axis=0)))


def wavelet_trans(args, x):
    """
    :param x: (batch, seq, N, channel)
    :return: xA3: (batch, N, channel, wavelet_seq)
    """
    batch, seq, N, channel = x.shape
    # get max level
    # w = pywt.Wavelet('db8') 
    # maxlev = pywt.dwt_max_level(seq_len, w.dec_len)
    
    x0 = np.empty((batch, N, channel, args.seq_len0), dtype=float)
    xA3 = np.empty((batch, N, channel, args.seq_len1), dtype=float)
    xD3 = np.empty((batch, N, channel, args.seq_len2), dtype=float)
    xD2 = np.empty((batch, N, channel, args.seq_len3), dtype=float)
    xD1 = np.empty((batch, N, channel, args.seq_len4), dtype=float)
    
    for i in range(batch):
        for j in range(N):
            for k in range(channel):
#                 cA3, cD3, cD2, cD1 = pywt.wavedec(x[i, :, j, k], 'db8', level=3)
#                 thresdata = x[i, :, j, k]
                #cA3, cD3, cD2, cD1 = pywt.wavedec(x[i, :, j, k], 'db8', level=3)
                #thresdata = ti(x[i, :, j, k])
                thresdata, cA3, cD3, cD2, cD1 = tsd(x[i, :, j, k])
                thresdata = tsd2(x[i, :, j, k])
                
                # check the length of wavelet transform:
                # print(len(cA3))
                # print(len(cD3))
                # print(len(cD2))
                # print(len(cD1))
                x0[i, j, k] = thresdata
                xA3[i, j, k] = cA3
                xD3[i, j, k] = cD3
                xD2[i, j, k] = cD2
                xD1[i, j, k] = cD1
    x0 = torch.from_numpy(x0).float()
    xA3 = torch.from_numpy(xA3).float()
    xD3 = torch.from_numpy(xD3).float()
    xD2 = torch.from_numpy(xD2).float()
    xD1 = torch.from_numpy(xD1).float()

    x0, xA3, xD3, xD2, xD1 = x0.permute(0, 3, 1, 2), xA3.permute(0, 3, 1, 2), xD3.permute(0, 3, 1, 2), \
                             xD2.permute(0, 3, 1, 2), xD1.permute(0, 3, 1, 2)

    return x0, xA3, xD3, xD2, xD1

def extract_factor(args, x):
    """
    :param x: batch, N, channel, seq
    :return: batch, N, channel + fac_num, seq
    """
    batch, N, channel, seq = x.shape
    out = torch.FloatTensor(batch, N, channel + args.factor_num, seq)
    for ba in range(batch):
        for i in range(N):
            # channel x seq
            X = x[ba, i, ...]
            # channel x channel
            cov = np.cov(X)
            # get first fac_num cov - > fac_num x fac_num
            val, vec = np.linalg.eig(cov)
            vec = torch.from_numpy(vec.astype(dtype=float)).float()
            V = vec[:args.factor_num].transpose(1, 0)
            V_t = vec[:args.factor_num]
            a = torch.mean(X, 1, True)
            one = torch.ones(seq).reshape(1, seq).float()
            x_a = X - a.mm(one)
            U = torch.matmul(torch.eye(channel).float() - V.mm(V_t), x_a)
            val = torch.from_numpy(val.astype(dtype=float)).float()
            lam = torch.diag(val[:args.factor_num])
            sq = 1 / torch.sqrt(lam)
            mask = torch.isinf(sq)
            sq = sq.masked_fill(mask, 0)
            F = x_a.t().mm(V).mm(sq)
            FM = torch.cat([U, F.t()], dim=0)
            out[ba, i] = FM
    return out


def get_var(cD):
    coeffs = cD
    abs_coeffs = []
    for coeff in coeffs:
        abs_coeffs.append(math.fabs(coeff))
    abs_coeffs.sort()
    pos = math.ceil(len(abs_coeffs) / 2)
    var = abs_coeffs[pos] / 0.6745
    return var

import math


def sure_shrink(var, coeffs):
    N = len(coeffs)
    sqr_coeffs = []
    for coeff in coeffs:
        sqr_coeffs.append(math.pow(coeff, 2))
    sqr_coeffs.sort()
    pos = 0
    r = 0
    for idx, sqr_coeff in enumerate(sqr_coeffs):
        new_r = (N - 2 * (idx + 1) + (N - (idx + 1))*sqr_coeff + sum(sqr_coeffs[0:idx+1])) / N
        if r == 0 or r > new_r:
            r = new_r
            pos = idx
    thre = math.sqrt(var) * math.sqrt(sqr_coeffs[pos])
    return thre



def visu_shrink(var, coeffs):
    N = len(coeffs)
    thre = math.sqrt(var) * math.sqrt(2 * math.log(N))
    return thre




def heur_sure(var, coeffs):
    N = len(coeffs)
    s = 0
    for coeff in coeffs:
        s += math.pow(coeff, 2)
    theta = (s - N) / N
    miu = math.pow(math.log2(N), 3/2) / math.pow(N, 1/2)
    if theta < miu:
        return visu_shrink(var, coeffs)
    else:
        return min(visu_shrink(var, coeffs), sure_shrink(var, coeffs))



def mini_max(var, coeffs):
    N = len(coeffs)
    if N > 32:
        return math.sqrt(var) * (0.3936 + 0.1829 * math.log2(N))
    else:
        return 0
    
    
def tsd(data, method='sureshrink', mode='soft', wavelets_name='db8', level=3):
#def tsd(data, method='minmax', mode='soft', wavelets_name='sym8', level=3):
    '''
    :param data: signal
    :param method: {'visushrink', 'sureshrink', 'heursure', 'minmax'}, 'sureshrink' as default
    :param mode: {'soft', 'hard', 'garotte', 'greater', 'less'}, 'soft' as default
    :param wavelets_name: wavelets name in PyWavelets, 'db8' as default
    :param level: deconstruct level, 3 as default
    :return: processed data
    '''
    methods_dict = {'visushrink': visu_shrink, 'sureshrink': sure_shrink, 'heursure': heur_sure, 'minmax': mini_max}

    wave = pywt.Wavelet(wavelets_name)



    data_ = data[:]
    (cA, cD) = pywt.dwt(data=data_, wavelet=wave)
    var = get_var(cD)
    coeffs = pywt.wavedec(data=data, wavelet=wavelets_name, level=level)
    
    for idx, coeff in enumerate(coeffs):
        if idx == 0:
            continue

        thre = methods_dict[method](var, coeff)

        coeffs[idx] = pywt.threshold(coeffs[idx], thre, mode=mode)
        

    thresholded_data = pywt.waverec(coeffs, wavelet=wavelets_name)

    return thresholded_data, coeffs[0], coeffs[1], coeffs[2], coeffs[3]


def tsd2(data, method='minmax', mode='soft', wavelets_name='db8', level=3): 
#def tsd2(data, method='minmax', mode='soft', wavelets_name='sym8', level=3): 
    '''
    :param data: signal
    :param method: {'visushrink', 'sureshrink', 'heursure', 'minmax'}, 'sureshrink' as default
    :param mode: {'soft', 'hard', 'garotte', 'greater', 'less'}, 'soft' as default
    :param wavelets_name: wavelets name in PyWavelets, 'db8' as default
    :param level: deconstruct level, 3 as default
    :return: processed data
    '''
    methods_dict = {'visushrink': visu_shrink, 'sureshrink': sure_shrink, 'heursure': heur_sure, 'minmax': mini_max}

    wave = pywt.Wavelet(wavelets_name)

 

    data_ = data[:]
    (cA, cD) = pywt.dwt(data=data_, wavelet=wave)
    var = get_var(cD)
    coeffs = pywt.wavedec(data=data, wavelet=wavelets_name, level=level)
    
    for idx, coeff in enumerate(coeffs):
        if idx == 0:
            continue

        thre = methods_dict[method](var, coeff)

        coeffs[idx] = pywt.threshold(coeffs[idx], thre, mode=mode)
        

    thresholded_data = pywt.waverec(coeffs, wavelet=wavelets_name)

    return thresholded_data


def right_shift(data, n):
    copy1 = list(data[n:])
    copy2 = list(data[:n])
    return copy1 + copy2




def back_shift(data, n):
    p = len(data) - n
    copy1 = list(data[p:])
    copy2 = list(data[:p])
    return copy1 + copy2


def ti(data, step=20, method='minmax', mode='soft', wavelets_name='db8', level=3):
#def ti(data, step=20, method='minmax', mode='soft', wavelets_name='sym8', level=3):
    '''
    :param data: signal
    :param step: shift step, 100 as default
    :param method: {'visushrink', 'sureshrink', 'heursure', 'minmax'}, 'heursure' as default
    :param mode: {'soft', 'hard', 'garotte', 'greater', 'less'}, 'soft' as default
    :param wavelets_name: wavelets name in PyWavelets, 'sym5' as default
    :param level: deconstruct level, 5 as default
    :return: processed data
    '''


    num = math.ceil(len(data)/step)
    final_data = [0]*len(data)
    for i in range(num):
        temp_data = right_shift(data, i*step)
        temp_data = tsd2(temp_data, method=method, mode=mode, wavelets_name=wavelets_name, level=level)
        #temp_data = temp_data.tolist()
        temp_data = temp_data.tolist() 
        temp_data = back_shift(temp_data, i*step)
        final_data = list(map(lambda x, y: x+y, final_data, temp_data))

    final_data = list(map(lambda x: x/num, final_data))
    return final_data


if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    args = util.get_common_args()
    args = args.parse_args()
    args.bidirectional = False
    print(args)

    cross_validation = True
    metrics = MetricCollector()
    if not cross_validation:
        datasets, scalers = load_dataset(args)
        t1 = time.time()
        training(args, datasets, scalers, metrics)
        print('total time cost', time.time() - t1)
        exit(1)

    # start cross_validation training:
    K = 5
    metrics = []
    base_dir = args.data_path
    for i in range(K):
        print(f"--------------------------start {i} fold training-----------------")
        mc = MetricCollector()
        args.data_path = os.path.join(base_dir, f'{i}/')
        datasets, scalers = load_dataset(args)
        t1 = time.time()
        #training(args, datasets, scalers, mc)
        training(args, datasets, scalers, mc, i)
        print('total time cost', time.time() - t1)
        metrics.append(mc)

    cross_validation_analysis(metrics)

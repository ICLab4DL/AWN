import wfdb as wfdb
import matplotlib.pyplot as plt
import numpy as np


class DictKey:
    def __init__(self, key):
        self.key = str(key)

    def __hash__(self):
        return self.key.__hash__()

    def __eq__(self, other):
        return self.__hash__().__eq__(other)


def draw_ecg(x):
    plt.plot(x)
    plt.title('Raw_ECG')
    plt.show()


def draw_ecg_R(record, annotation):
    plt.plot(record.p_signal)  
    R_v = record.p_signal[annotation.sample]  
    plt.plot(annotation.sample, R_v, 'or')  
    plt.title('Raw_ECG And R Position')
    plt.show()


def selData(record, annotation, label, R_left, R_right):
    a = annotation.symbol
    d = dict()
    b = [a[k] for k in range(len(a))]
    for i in b:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    f = [k for k in range(len(a)) if a[k] == label]  
    signal = record.p_signal
    R_pos = annotation.sample[f]
    res = []
    for i in range(len(f)):
        if (R_pos[i] - R_left > 0) and (len(signal) - R_pos[i]) > R_right:
            res.append(signal[R_pos[i] - R_left:R_pos[i] + R_right + 1])
    return res


# 读取心电图数据
def read_ecg_data(filePath, channel_names):
 

    record = wfdb.rdrecord(filePath, channel_names=channel_names)
   
    signame = record.sig_name
    if signame is None or (len(signame) != 2) or ("MLII" not in signame) or ("V1" not in signame):
        print("not both:", record.sig_name)
        return None, None
    print(record.sig_name)  
    # print(record.sig_len)

    annotation = wfdb.rdann(filePath, 'atr')
    #     print(annotation.symbol)
    return record, annotation


import os


symbs = ["N", "L", "R", "A", "V"]

num_symbs = len(symbs)

# K-folder
K = 5

def generate_data():
    all_records = []
    left = 269 
    right = 200
    total = left + right + 1
    baseDir = './mit-bih-arrhythmia-database-1.0.0/'
    for i in range(100, 235):
        filePath = baseDir + str(i)
        if not os.path.exists(filePath + ".hea"):
            continue

        Channel_Name = ['MLII', 'V1']
        record, annotation = read_ecg_data(filePath, Channel_Name)
        if record is None:
            continue
        all_records.append((record, annotation))
    #     draw_ecg(record.p_signal)
    # draw_ecg_R(record, annotation)
    ress = dict()
    for s in symbs:
        res = []
        for i in range(len(all_records)):
            record, annotation = all_records[i]
            if record.sig_name is None:
                continue
            data = np.array(selData(record, annotation, s, left, right))
            if len(data) < 1:
                continue
            if data.shape[1] != total or data.shape[2] != 2:
                print("shape::::::::: ", data.shape)
                print("iiiiiiiii: ", i)
            # data = np.array(selData(record, annotation, s, left, right)).squeeze().transpose(1, 0)
            res.append(data)
        res_stack = np.concatenate(res, axis=0)
        ress[s] = res_stack
    return ress


def cross_validation_slicing(K, x_data, y_data):
    """
    simple cross validation tool
    Parameters
    ----------
    K:        K-folder
    x_data:   all samples
    y_data:   all labels

    Returns:  K x (train(x, y), test(x, y), val(x, y))
    -------
    """

    def exclude_target(data, target):
        """
        combine and put all the slices from data except the target slice.
        """
        slices = []
        for i in range(len(data)):
            if i != target:
                slices.append(data[i])
        return slices

    print(x_data.shape)
    np.random.shuffle(x_data)
    num_samples = x_data.shape[0]
    num_each = round(num_samples / K)
    if num_each < 1:
        raise ValueError('number sections must be larger than 0.')
    x_splits = np.array_split(x_data, K, axis=0)
    y_splits = np.array_split(y_data, K, axis=0)

    train = []
    test = []
    val = []
    for i in range(K):
        test.append((x_splits[i], y_splits[i]))
        if i < K - 1:
            val.append((x_splits[i+1], y_splits[i+1]))
        else:
            val.append((x_splits[0], y_splits[0]))
        slices_x = exclude_target(x_splits, i)
        slices_y = exclude_target(y_splits, i)
        train.append((np.concatenate(slices_x[:], axis=0), np.concatenate(slices_y[:], axis=0)))
    print("train[0][0]", train[0][0].shape)
    print("test[0][0]", test[0][0].shape)
    print("val[0][0]", val[0][0].shape)
    return train, test, val


def save_data(data):
    label = 0

    # do the K-fold cross validation slicing for each category:
    trains = []
    tests = []
    vals = []
    for k, v in data.items():
        samples = v.shape[0]
        train, test, val = cross_validation_slicing(K, v, np.array([label for j in range(samples)]))
        trains.append(train)
        tests.append(test)
        vals.append(val)
        label += 1

    # ----------- combine done ---------------------------
    # combine all the categories:
    train_sets = []
    test_sets = []
    val_sets = []
    category_num = num_symbs
    for i in range(K):
        each_trains_x = []
        each_tests_x = []
        each_vals_x = []
        each_trains_y = []
        each_tests_y = []
        each_vals_y = []
        for j in range(category_num):
            each_trains_x.append(trains[j][i][0])
            each_tests_x.append(tests[j][i][0])
            each_vals_x.append(vals[j][i][0])
            each_trains_y.append(trains[j][i][1])
            each_tests_y.append(tests[j][i][1])
            each_vals_y.append(vals[j][i][1])

        train_sets.append((np.concatenate(each_trains_x, axis=0), np.concatenate(each_trains_y, axis=0)))
        test_sets.append((np.concatenate(each_tests_x, axis=0), np.concatenate(each_tests_y, axis=0)))
        val_sets.append((np.concatenate(each_vals_x, axis=0), np.concatenate(each_vals_y, axis=0)))


    # save as K files:
    dir = "260all_crossval"
    for i in range(K):
        path = f"./{dir}/{i}/"
        if os.path.exists(path):
            reply = str(input(f'{path}train.npz exists. Do you want to overwrite it? (y/n)')).lower().strip()
            if reply[0] != 'y': exit
        else:
            os.makedirs(path)

        np.savez_compressed(
            os.path.join(path, "train.npz"),
            x=train_sets[i][0],
            y=train_sets[i][1],
        )
        np.savez_compressed(
            os.path.join(path, "test.npz"),
            x=test_sets[i][0],
            y=test_sets[i][1],
        )
        np.savez_compressed(
            os.path.join(path, "val.npz"),
            x=val_sets[i][0],
            y=val_sets[i][1],
        )


def load_data():
    train = np.load("heart_data/train.npz", allow_pickle=True)
    test = np.load("heart_data/test.npz", allow_pickle=True)
    val = np.load("heart_data/val.npz", allow_pickle=True)
    print(train["x"].shape)
    print(train["y"].shape)

    print(test["x"].shape)
    print(val["x"].shape)

    # plot:
    # for i in range(10):
    #     plt.figure(figsize=(20, 8))
    #     plt.plot(np.arange(0, total), res[i, :, 0])
    #     plt.plot(np.arange(0, total), res[i, :, 1])
    #     plt.show()


if __name__ == "__main__":
    data = generate_data()
    save_data(data)

    # load_data()

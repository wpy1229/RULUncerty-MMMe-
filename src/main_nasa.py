import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers

from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from model import * 
from early_stopping import * 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# convert str to datatime 
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# load .mat data
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split("/")[-1].split(".")[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data


# get capacity data
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]


# get the charge data of a battery
def getBatteryValues(Battery, Type='charge'):
    data=[]
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data


Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = '../datasets/NASA/'

Battery = {}
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    Battery[name] = getBatteryCapacity(data)


# Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
# Battery = np.load('datasets/NASA/NASA.npy', allow_pickle=True)
# Battery = Battery.item()
fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b:', 'g--', 'r-.', 'c.']
c = 0
for name,color in zip(Battery_list, color_list):
    df_result = Battery[name]
    ax.plot(df_result[0], df_result[1], color, label=name)
ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
plt.legend()
plt.savefig('../figures/nasa_demo.png')


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)
        
    return np.array(x), np.array(y)


def split_dataset(data_sequence, train_ratio=0.0, capacity_threshold=0.0):
    if capacity_threshold > 0:
        max_capacity = max(data_sequence)
        capacity = max_capacity * capacity_threshold
        point = [i for i in range(len(data_sequence)) if data_sequence[i] < capacity]
    else:
        point = int(train_ratio + 1)
        if 0 < train_ratio <= 1:
            point = int(len(data_sequence) * train_ratio)
    train_data, test_data = data_sequence[:point], data_sequence[point:]
    return train_data, test_data


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, valid_name,   window_size=8):
    data_sequence=data_dict[name][1]
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    data_sequence=data_dict[name][1]
    train_valid_data, valid_data = data_sequence[:window_size+1], data_sequence[window_size+1:]

    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v[1], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data), list(train_valid_data), list(valid_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re)/true_re


def evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    mae = mean_absolute_error(y_test, y_predict)
    return rmse, mae
    
    
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

 

def train( ):
    score_list, result_list = [], []
    setup_seed(seed)
    for i in range(4):
        name = Battery_list[i]
        if i + 1 == 4:
            valid_name = Battery_list[0]
        else:
            valid_name = Battery_list[i+1]
        window_size = feature_size
        train_x, train_y, train_data, test_data, train_valid_data, valid_data = get_train_test(Battery, name, valid_name, window_size)
        train_size = len(train_x)
        # print('sample size: {}'.format(train_size))

        model = Net(feature_size)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
        criterion = nn.MSELoss()
        stopper = EarlyStopping(mode='lower', patience=200, filename='models/nasa')


        test_x = train_data.copy()
        loss_list, y_ = [0], []
        rmse, re = 1, 1
        score_, score = [1],[1]
        for epoch in range(EPOCH):
            model.train()
            X = np.reshape(train_x/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32) # (batch_size, seq_len, input_size)
            y = np.reshape(train_y[:,-1]/Rated_Capacity,(-1,1)).astype(np.float32)          # (batch_size, 1)

            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            output = model(X)
            output = output.reshape(-1, 1)
            loss = criterion(output, y)  
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            # val
            y_ = []
            model.eval()
            test_x = train_valid_data.copy() 
            point_list = []
            while (len(test_x) - len(train_valid_data)) < len(valid_data):
                x = np.reshape(np.array(test_x[-feature_size:])/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
                x = torch.from_numpy(x).to(device)   # shape (batch_size,feature_size=1,input_size)
                pred = model(x)                   # pred shape: (batch_size=1, feature_size=1)
                next_point = pred.data.cpu().numpy()[0,0] * Rated_Capacity
                test_x.append(next_point)      # The test values are added to the original sequence to continue to predict the next point
                point_list.append(next_point)  # Saves the predicted value of the last point in the output sequence
            y_.append(point_list)              # Save all the predicted values

            loss_list.append(loss)
            rmse, mae = evaluation(y_test=valid_data, y_predict=y_[-1])
            re = relative_error(y_test=valid_data, y_predict=y_[-1], threshold=Rated_Capacity*0.7)
            print('epoch:{:<2d} | loss:{:<6.4f} | RMSE:{:<6.4f} |MAE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, rmse, mae, re))
            early_stop = stopper.step(re, model)
            if early_stop:
                break
        

             
        stopper.load_checkpoint(model)
        y_ = []
        model.eval()
        test_x = train_data.copy() 
        point_list = []
        while (len(test_x) - len(train_data)) < len(test_data):
            x = np.reshape(np.array(test_x[-feature_size:])/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
            x = torch.from_numpy(x).to(device)   # shape (batch_size,feature_size=1,input_size)
            pred = model(x)                   # pred shape: (batch_size=1, feature_size=1)
            next_point = pred.data.cpu().numpy()[0,0] * Rated_Capacity
            test_x.append(next_point)      # The test values are added to the original sequence to continue to predict the next point
            point_list.append(next_point)  # Saves the predicted value of the last point in the output sequence
        y_.append(point_list)              # Save all the predicted values

        loss_list.append(loss)
        rmse, mae = evaluation(y_test=test_data, y_predict=y_[-1])
        re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity*0.7)

        # np.savez('data/nasa'+name, true=test_data, pred=y_[-1])

        
        print(' Test--------- | loss:{:<6.4f} | RMSE:{:<6.4f} |MAE:{:<6.4f} | RE:{:<6.4f}'.format( loss, rmse, mae, re))
        if metric == 're':
            score = [re]
        elif metric == 'rmse':
            score = [rmse]
        elif metric == 'mae':
            score = [mae]
        else:
            score = [re, rmse, mae]
            
        score_ = score.copy()
            
        score_list.append(score_)
        result_list.append(y_[-1])
    return score_list, result_list

Rated_Capacity = 2.0
window_size = 16
feature_size = window_size
dropout = 0.0
EPOCH = 1000
nhead = 8
hidden_dim = 16
num_layers = 1
lr = 1e-2   # learning rate
weight_decay = 0.0
noise_level = 0.0
alpha = 1e-5
is_load_weights = False
metric = 'all'
seed = 0

 


SCORE = []
print('seed:{}'.format(seed))
score_list, _ = train()
 
print(np.array(score_list))

print(np.mean(score_list, axis=0))
# for s in score_list:
#     SCORE.append(s)

# print('------------------------------------------------------------------')
# print(metric + ' mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))
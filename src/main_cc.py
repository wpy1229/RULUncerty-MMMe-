import numpy as np
import random
import math
import time
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import * 
from early_stopping import * 
from sklearn.model_selection import train_test_split

from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def build_sequences(text, window_size):
    #text:list of capacity
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        target = text[i+1:i+1+window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
# def get_train_test(data_dict, name, window_size=8):
#     data_sequence=data_dict[name]['capacity']
#     train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
#     train_x, train_y = build_sequences(text=train_data, window_size=window_size)
#     for k, v in data_dict.items():
#         if k != name:
#             data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
#             train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
#     return train_x, train_y, list(train_data), list(test_data)


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, valid_name,   window_size=8):
    data_sequence=data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    data_sequence=data_dict[name]['capacity']
    train_valid_data, valid_data = data_sequence[:window_size+1], data_sequence[window_size+1:]

    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
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
    return abs(true_re - pred_re)/true_re if abs(true_re - pred_re)/true_re<=1 else 1


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


# Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']

# dir_path = '../datasets/CALCE/'
# Battery = {}
# for name in Battery_list:
#     print('Load Dataset ' + name + ' ...')
#     path = glob.glob(dir_path + name + '/*.xlsx')
#     dates = []
#     for p in path:
#         df = pd.read_excel(p, sheet_name=1)
#         print('Load ' + str(p) + ' ...')
#         dates.append(df['Date_Time'][0])
#     idx = np.argsort(dates)
#     path_sorted = np.array(path)[idx]
    
#     count = 0
#     discharge_capacities = []
#     health_indicator = []
#     internal_resistance = []
#     CCCT = []
#     CVCT = []
#     for p in path_sorted:
#         df = pd.read_excel(p,sheet_name=1)
#         print('Load ' + str(p) + ' ...')
#         cycles = list(set(df['Cycle_Index']))
#         for c in cycles:
#             df_lim = df[df['Cycle_Index'] == c]
#             #Charging
#             df_c = df_lim[(df_lim['Step_Index'] == 2)|(df_lim['Step_Index'] == 4)]
#             c_v = df_c['Voltage(V)']
#             c_c = df_c['Current(A)']
#             c_t = df_c['Test_Time(s)']
#             #CC or CV
#             df_cc = df_lim[df_lim['Step_Index'] == 2]
#             df_cv = df_lim[df_lim['Step_Index'] == 4]
#             CCCT.append(np.max(df_cc['Test_Time(s)'])-np.min(df_cc['Test_Time(s)']))
#             CVCT.append(np.max(df_cv['Test_Time(s)'])-np.min(df_cv['Test_Time(s)']))

#             #Discharging
#             df_d = df_lim[df_lim['Step_Index'] == 7]
#             d_v = df_d['Voltage(V)']
#             d_c = df_d['Current(A)']
#             d_t = df_d['Test_Time(s)']
#             d_im = df_d['Internal_Resistance(Ohm)']

#             if(len(list(d_c)) != 0):
#                 time_diff = np.diff(list(d_t))
#                 d_c = np.array(list(d_c))[1:]
#                 discharge_capacity = time_diff*d_c/3600 # Q = A*h
#                 discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(discharge_capacity.shape[0])]
#                 discharge_capacities.append(-1*discharge_capacity[-1])

#                 dec = np.abs(np.array(d_v) - 3.8)[1:]
#                 start = np.array(discharge_capacity)[np.argmin(dec)]
#                 dec = np.abs(np.array(d_v) - 3.4)[1:]
#                 end = np.array(discharge_capacity)[np.argmin(dec)]
#                 health_indicator.append(-1 * (end - start))

#                 internal_resistance.append(np.mean(np.array(d_im)))
#                 count += 1

#     discharge_capacities = np.array(discharge_capacities)
#     health_indicator = np.array(health_indicator)
#     internal_resistance = np.array(internal_resistance)
#     CCCT = np.array(CCCT)
#     CVCT = np.array(CVCT)
    
#     idx = drop_outlier(discharge_capacities, count, 40)
#     df_result = pd.DataFrame({'cycle':np.linspace(1,idx.shape[0],idx.shape[0]),
#                               'capacity':discharge_capacities[idx],
#                               'SoH':health_indicator[idx],
#                               'resistance':internal_resistance[idx],
#                               'CCCT':CCCT[idx],
#                               'CVCT':CVCT[idx]})
#     Battery[name] = df_result

Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('../datasets/CALCE/CALCE.npy', allow_pickle=True)
Battery = Battery.item()

def train():
    score_list, result_list = [], []
    
    for i in range(4):
        name = Battery_list[i]
        window_size = feature_size
        # train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size)

        if i + 1 == 4:
            valid_name = Battery_list[0]
        else:
            valid_name = Battery_list[i+1]
        window_size = feature_size
        train_x, train_y, train_data, test_data, train_valid_data, valid_data = get_train_test(Battery, name, valid_name, window_size)



        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        setup_seed(seed)
        model = Net(window_size)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
        criterion = nn.MSELoss()
        stopper = EarlyStopping(mode='lower', patience=200, filename='models/cc')

 
        test_x = train_data.copy()
        loss_list, y_ = [0], []
        rmse, re = 1, 1
        score_, score = [1], [1]


        # train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
 
        for epoch in range(EPOCH):
            X = np.reshape(train_x/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)#(batch_size, seq_len, input_size)
            y = np.reshape(train_y[:,-1]/Rated_Capacity,(-1,1)).astype(np.float32)# shape 为 (batch_size, 1)

            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            model.train()
            output = model(X)
            output = output.reshape(-1, 1)
            loss = criterion(output, y) 
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

            #  valid
             
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
            x = torch.from_numpy(x).to(device) # (batch_size,feature_size=1,input_size)
            
            pred  = model(x)                 # pred shape (batch_size=1, feature_size=1)
            next_point = pred.data.cpu().numpy()[0,0] * Rated_Capacity
            test_x.append(next_point)     # The test values are added to the original sequence to continue to predict the next point
            point_list.append(next_point) # Saves the predicted value of the last point in the output sequence
        y_.append(point_list)             # Save all the predicted values
        loss_list.append(loss)
        rmse, mae = evaluation(y_test=test_data, y_predict=y_[-1])
        re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity*0.7)
        np.savez('data/cc'+name, true=test_data, pred=y_[-1])

        print('epoch:{:<2d} | loss:{:<6.4f} | RMSE:{:<6.4f} |MAE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, rmse, mae, re))
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


Rated_Capacity = 1.1
window_size = 64
feature_size = window_size
dropout = 0.0
EPOCH = 1000
nhead = 16
weight_decay = 0.0
noise_level = 0.0
alpha = 0.01
lr = 0.0005    # learning rate
hidden_dim = 32
num_layers = 1
is_load_weights = True
metric = 'all'

seed = 0
SCORE = []
print('seed:{}'.format(seed))
score_list, _ = train()
print(np.array(score_list))
print(np.mean(score_list, axis=0))

# print(metric + ': {:<6.4f}'.format(np.mean(np.array(score_list))))
# print('------------------------------------------------------------------')
# for s in score_list:
#     SCORE.append(s)

# print(metric + ' mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))
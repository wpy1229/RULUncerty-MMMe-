import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime
import scienceplots

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


Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('../datasets/CALCE/CALCE.npy', allow_pickle=True)
Battery = Battery.item()

fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b:', 'g--', 'r-.', 'c.']
for name,color in zip(Battery_list, color_list):
    df_result = Battery[name]
    ax.plot(df_result['cycle'], df_result['capacity'], color, label='Battery_'+name)
ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 1°C')
plt.legend()
Rated_Capacity = 1.1
with plt.style.context(['science','no-latex']):
    name = 'CS2_35'
    fig, ax = plt.subplots()
    df_result = Battery[name]
    pred_results = np.load('data/cc'+name+'.npz')
    pred_results_trm = np.load('data/calce_trm'+name+'.npz')
    ax.plot(df_result['cycle'], df_result['capacity'], 'r', label='Real')
    ax.plot(df_result['cycle'][65:], pred_results['pred'], 'k:', label='MMMe', linewidth=2)
    ax.plot(df_result['cycle'][65:], pred_results_trm['pred'], 'g:', label='DeTransformer', linewidth=2)
    ax.vlines(x = 65, ymin = 0.8, ymax = 1.2,
           colors = 'b', linestyle = '--')
    ax.axhline(y = Rated_Capacity*0.7, color = 'k', linestyle = '--')
    text_y = 1.1
    ax.text(68, text_y, "Prediction start point=65", fontsize=10, color = "k"   )
    ax.text(10, Rated_Capacity*0.7-0.15, "Failure threshold", fontsize=10, color = "k")

    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)')
    ax.autoscale(tight=True)
    ax.legend(loc='lower left')
    plt.ylim(0, 1.25)
    plt.savefig('../figures/'+name+'pred.png',dpi=300)





# Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
# dir_path = '../datasets/NASA/'
#
# Battery = {}
# for name in Battery_list:
#     print('Load Dataset ' + name + '.mat ...')
#     path = dir_path + name + '.mat'
#     data = loadMat(path)
#     Battery[name] = getBatteryCapacity(data)
#
#
# color_list = ['r', 'g--', 'r-.', 'c.']
# c = 0
# with plt.style.context(['science','no-latex']):
#     name = 'B0007'
#     fig, ax = plt.subplots()
#     df_result = Battery[name]
#     pred_results = np.load('data/nasa'+name+'.npz')
#     pred_results_trm = np.load('data/nasa_trm'+name+'.npz')
#     ax.plot(df_result[0], df_result[1], 'r', label='Real')
#     ax.plot(df_result[0][17:], pred_results['pred'], 'k:', label='MMMe', linewidth=2)
#     ax.plot(df_result[0][17:], pred_results_trm['pred'], 'g:', label='DeTransformer', linewidth=2)
#     ax.vlines(x = 17, ymin = 1.7, ymax = 2.0,
#            colors = 'b', linestyle = '--')
#     ax.axhline(y = 1.4, color = 'k', linestyle = '--')
#     text_y = 1.9
#     ax.text(17, text_y, "Prediction start point=17", fontsize=10, color = "k"   )
#     ax.text(5, 1.3, "Failure threshold", fontsize=10, color = "k")
#
#     ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)')
#     ax.autoscale(tight=True)
#     # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
#     #       fancybox=True, shadow=True, ncol=3)
#     ax.legend(bbox_to_anchor=(-0.2, 1.02, 1.2, 0.2), loc="lower left",
#                 mode="expand", borderaxespad=0, ncol=3)
#     # loc='upper center', bbox_to_anchor=(0.5, -0.05),
#     #       fancybox=True, shadow=True, ncol=5
#     plt.ylim(1.2, 2.1)
#     plt.savefig('../figures/'+name+'pred.png',dpi=300)

# for name,color in zip(Battery_list, color_list):
#     df_result = Battery[name]
#     print(df_result[1], df_result[0])
#     ax.plot(df_result[0], df_result[1], color, label=name)
# ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
# plt.legend()
# plt.savefig('../figures/nasa_demo.png')

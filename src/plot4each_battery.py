import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
from datetime import datetime
import scienceplots

Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
Battery = np.load('../datasets/CALCE/CALCE.npy', allow_pickle=True)
Battery = Battery.item()
with plt.style.context(['science','no-latex']):
    fig, ax = plt.subplots()
    color_list = ['b:', 'g--', 'r-.', 'c--']
    for name,color in zip(Battery_list, color_list):
        df_result = Battery[name]
        ax.plot(df_result['cycle'], df_result['capacity'], color, label=name)
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)')
    ax.axhline(y = 1.1*0.7, color = 'k', linestyle = '--')
    ax.text(5,  0.69, "Failure threshold", fontsize=10, color = "k")
    plt.legend()
    plt.savefig('../figures/cacle.png',dpi=300)

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
Battery = np.load('../datasets/NASA/NASA.npy', allow_pickle=True)
Battery = Battery.item()

with plt.style.context(['science','no-latex']):
    fig, ax = plt.subplots()
    color_list = ['b:', 'g--', 'r-.', 'c--']
    c = 0
    for name,color in zip(Battery_list, color_list):
        df_result = Battery[name]
        ax.plot(df_result[0], df_result[1], color, label=name)
    ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)')
    ax.axhline(y = 1.4, color = 'k', linestyle = '--')
    ax.text(5, 1.3, "Failure threshold", fontsize=10, color = "k")
    plt.legend()
    plt.savefig('../figures/nasa.png',dpi=300)

# def relative_error(y_test, y_predict, threshold):
#     true_re, pred_re = len(y_test), 0
#     for i in range(len(y_test)-1):
#         if y_test[i] <= threshold >= y_test[i+1]:
#             true_re = i - 1
#             break
#     for i in range(len(y_predict)-1):
#         if y_predict[i] <= threshold:
#             pred_re = i - 1
#             break
#     return abs(true_re - pred_re)/true_re


# def evaluation(y_test, y_predict):
#     mse = mean_squared_error(y_test, y_predict)
#     rmse = sqrt(mean_squared_error(y_test, y_predict))
#     mae = mean_absolute_error(y_test, y_predict)
#     return rmse, mae

# def compute_metrics(file_name, Rated_Capacity):
#     pred_results = np.load('data/'+file_name+'.npz')
#     rmse, mae = evaluation(y_test=pred_results['true'], y_predict=pred_results['pred'])
#     re = relative_error(y_test=pred_results['true'], y_predict=pred_results['pred'], threshold=Rated_Capacity*0.7)
#     return rmse, mae, re


# x = [4, 8, 16, 32, 64]
# y1 = [ ]
# y2 = [ ]
# y3 = [ ]

# batteries = ("CS2_35", "CS2_36", "CS2_37","CS2_38")
# metrics = {
#     'RE':[],
#     'RMSE':[],
#     'MAE': [],
# }
# for name in batteries:
#     rmse, mae, re = compute_metrics('cc'+name, Rated_Capacity = 1.1)
#     metrics['RE'].append(re)
#     metrics['RMSE'].append(rmse)
#     metrics['MAE'].append(mae)
    

# with plt.style.context(['science','no-latex']):
     
#     x = np.arange(len(batteries))  # the label locations
#     width = 0.25  # the width of the bars
#     multiplier = 0

#     fig, ax = plt.subplots(layout='constrained')

#     for attribute, measurement in metrics.items():
#         offset = width * multiplier
#         rects = ax.bar(x + offset, measurement, width, label=attribute)
#         ax.bar_label(rects, padding=3)
#         multiplier += 1

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Length (mm)')
     
#     ax.set_xticks(x + width, species)
#     ax.legend(loc='upper left', ncols=3)

       
     
    # plt.savefig('../figures/'+name+'pred.png')
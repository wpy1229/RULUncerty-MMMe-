# [0.00496689 0.05895844 0.04563775] 4 experts
# [0.00993377 0.0578618  0.04646758] 8 experts
# [0.0115894  0.05130112 0.04192977] 16 experts
# [0.005 0.0515 0.04] 32 experts
# [0.0325791  0.05815233 0.045674  ] 64
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
with plt.style.context(['science','no-latex']):
    fig, ax1 = plt.subplots()
    x = [4, 8, 16, 32, 64]
    y1 = [0.00496689, 0.00993377,0.0115894, 0.005, 0.0325791]
    y2 = [0.05895844, 0.0578618,  0.05130112, 0.0515, 0.05815233]
    y3 = [0.04563775, 0.04646758, 0.04192977, 0.04, 0.045674 ]
    line1 = ax1.plot(x,y1, '-s')
    ax1.set_ylabel('RE', color='C0')
    ax1.set_xlabel('# of Experts')
    ax1.tick_params(axis='y', color='C0', labelcolor='C0')
    ax1.set_xticks(x)

    # ax1.set_title('sin(x) and exp(x)')

    ax2 = ax1.twinx()
    line2 = ax2.plot(x,y2,'-o',color='C1')
    ax2.set_ylabel('RMSE', color='C1')
    ax2.tick_params(axis='y', color='C1', labelcolor='C1')
    # ax2.spines['right'].set_color('C1')
    # ax2.spines['left'].set_color('C0')

    ax3 = ax1.twinx()

    line3 = ax3.plot(x,y3,'-^',color='C2')
    ax3.set_ylabel('MAE', color='C2')
    ax3.tick_params(axis='y', color='C2', labelcolor='C2')
     
    ax1.yaxis.label.set_color(color='C0')
    ax2.yaxis.label.set_color(color='C1')
    ax3.yaxis.label.set_color(color='C2')
    ax3.spines.right.set_position(("axes", 1.3))
    lines = line1 + line2 + line3
    ax2.legend(lines, ['RE','RMSE', 'MAE'])

    plt.savefig('../figures/yy_moe_nasa.png',dpi=300)
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.linalg import pinv
from Main_Chebyshev5input import WASDSSECI_5inputwnh
import matplotlib.pyplot as plt

def Predictions_data():
    rate_test = 0.01
    start_date = '01/03/2023'
    end_date = '06/30/2024'
    sseci = pd.read_excel('上证指数.xlsx')
    sseci['时间'] = pd.to_datetime(sseci['时间'], format='%Y-%m-%d')
    sseci = sseci.set_index('时间')['收盘价(元)'].loc[start_date:end_date]
    time_ls = sseci.index.strftime('%m/%d/%Y').tolist()  # 转换为列表
    Divs=[]
    Predictions=[]
    for i in range(len(time_ls)):
        zyntime, Predict_price, entire_t, actual_p, entire_FuncValue=WASDSSECI_5inputwnh(rate_test, time_ls[i])
        div=np.abs((Predict_price-actual_p)/actual_p)
        Divs.append(div)
        Predictions.append(Predict_price)
    pd.DataFrame(Predictions, columns=['Predictions']).to_csv('train/Predictions_plot.csv', index=False)
    pd.DataFrame(Divs, columns=['Divergence']).to_csv('train/Divs_plot.csv', index=False)

if __name__ == "__main__":
    # Predictions_data()
    rate_test = 0.01
    Predictions = pd.read_csv('train/Predictions_plot.csv')['Predictions'].values
    Divs = pd.read_csv('train/Divs_plot.csv')['Divergence'].values
    start_date = '01/03/2023'
    end_date = '06/30/2024'
    sseci = pd.read_excel('上证指数.xlsx')
    sseci['时间'] = pd.to_datetime(sseci['时间'], format='%Y-%m-%d')
    sseci = sseci.set_index('时间')['收盘价(元)'].loc[start_date:end_date]
    time_ls = sseci.index.strftime('%m/%d/%Y').tolist()  # 转换为列表

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    ax.plot(time_ls, Divs)
    ax.set_xlabel('Date')
    ax.set_ylabel('Difference Percentage')
    ax.set_xticks(time_ls[::50])
    fig.tight_layout()
    fig.savefig('plot/errors_for_chebyshev.pdf')
    plt.close(fig)

    # 绘制价格图
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 8)
    ax.plot(time_ls, Predictions, linestyle='--', color='red', label='Predicted Price')
    ax.plot(time_ls, sseci, label='Actual Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_xticks(time_ls[::50])
    ax.legend()

    # 添加小图（放大图）
    axins = ax.inset_axes((0.1, 0.1, 0.3, 0.3))
    axins.plot(time_ls, Predictions, linestyle='--', color='red', label='Predicted Price')
    axins.plot(time_ls, sseci,label='Actual Price')
    axins.set_xticks(time_ls[::7])
    # 设置放大区间
    zone_left = "10/09/2023"
    zone_right = "11/16/2023"
    x_ratio = 0  # x轴显示范围的扩展比例
    y_ratio = 0.05  # y轴显示范围的扩展比例

    # 获取放大区间的索引
    left_idx = time_ls.index(zone_left)
    right_idx = time_ls.index(zone_right)

    # X轴的显示范围
    xlim0 = left_idx - (right_idx - left_idx) * x_ratio
    xlim1 = right_idx + (right_idx - left_idx) * x_ratio

    # Y轴的显示范围
    y = np.hstack((Predictions[left_idx:right_idx], sseci.values[left_idx:right_idx]))
    ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # 在原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")
    print(sx)
    print(sy)


    plt.tight_layout()
    plt.savefig('plot/price_for_chebyshev.pdf')
    plt.close()
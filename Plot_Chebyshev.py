import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import yaml

def plot_Process():
    """
    Plots the actual price of the Shanghai Stock Exchange Composite Index (SSECI) along with buy and sell signals.
    This function reads the actual closing prices from an Excel file and buy/sell signals from CSV files.
    It then calculates the difference between predicted and actual prices, and plots the actual prices,
    buy/sell signals, and the differences on a dual-axis plot.
    The plot is saved as a PDF file.
    Parameters:
    None
    Returns:
    None
    Notes:
    - The function assumes the presence of specific files in the 'train' directory:
      'buy_signal_predict.csv' and 'sell_signal_predict.csv'.
    - The actual prices are read from '上证指数.xlsx' file.
    - The Date range for plotting is hardcoded from '2024-01-01' to '2024-12-31'.
    - The function saves the plot as 'plot/ASF_predict_CHEBYSHEV.pdf'.
    """
    with open('para.yml') as f:
        settings = yaml.safe_load(f)
    year = settings['year']
    file_path = '上证指数.xlsx'
    # start_date = f'01/04/{year}'
    # end_date = f'12/30/{year}'
    wbuy = pd.read_csv(f'train/buy_signal_predict_{year}.csv')
    wsell = pd.read_csv(f'train/sell_signal_predict_{year}.csv')
    sseci = pd.read_excel(file_path)
    
    sseci['year'] = sseci['时间'].dt.year
    sseci = sseci[sseci['year'] == year]

    sseci['date'] = pd.to_datetime(sseci['时间'])
    sseci['date']=sseci['date'].dt.strftime('%m/%d/%Y')
    sseci = sseci.set_index('date')['收盘价(元)']
    # 2024-09-30
    # 合并买入和卖出信号
    wbuy['type'] = 'buy'
    wsell['type'] = 'sell'
    signals = pd.concat([wbuy, wsell])
    
    # 计算差值
    signals['difference'] = signals['Predict_price'] - signals['actual_p']
    signals.to_csv(f'train/signals_{year}.csv', index=False)
    
    # 创建画布和主 Y 轴
    fig, ax1 = plt.subplots(figsize=(12, 8))
    L_buy = len(wbuy)
    L_sell = len(wsell)
    # 绘制实际价格曲线（主 Y 轴）
    ax1.plot(sseci, label='Actual Price')
    ax1.scatter(wbuy['Date'], sseci.loc[wbuy['Date']], color='green', marker='v', label='Buy Signal', s=20)
    ax1.scatter(wsell['Date'], sseci.loc[wsell['Date']], color='red', marker='^', label='Sell Signal', s=20)
    # ax1.scatter(signals['Date'], signals['Predict_price'], label='Predict Price')
    ax1.set_xlabel('Date')
    ax1.set_xticks(np.arange(0, len(sseci), 50))
    ax1.set_ylabel('Actual Price')
    ax1.tick_params(axis='y')
    plt.tight_layout()
    
    # 创建副 Y 轴
    ax2 = ax1.twinx()
    # 绘制买入和卖出信号的柱状图（副 Y 轴）
    for i, row in signals.iterrows():
        if row['type'] == 'buy':
            color = 'green'  # 阳线
        else:
            color = 'red'  # 阴线
        print(row['Date'], row['difference'])
        ax2.bar(row['Date'], row['difference'],color=color, width=3)
    
    ax2.set_ylabel('Difference (Predict - Actual)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    # 添加图例
    fig.legend()
    
    plt.tight_layout()
    plt.savefig(f'plot/ASF_predict_CHEBYSHEV_{year}.pdf')

if __name__ == '__main__':
    plot_Process()
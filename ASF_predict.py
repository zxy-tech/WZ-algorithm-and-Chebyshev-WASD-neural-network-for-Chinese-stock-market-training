import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import yaml
def Vs_N_Ps(file_path, start_date='12/19/1990', end_date='12/29/2023', Days=60):
    sseci = pd.read_excel(file_path)
    sseci['时间'] = pd.to_datetime(sseci['时间'], format='%Y-%m-%d')
    sseci = sseci.set_index('时间')['收盘价(元)'].loc[start_date:end_date]
    ls_p = {}
    ls_v = {}
    
    # 遍历数据，寻找局部峰值
    for i in range(Days, len(sseci) - Days):  # 从第 31 个数据开始，到倒数第 31 个数据结束
        formatted_time = sseci.index[i].strftime('%m/%d/%Y')
        # 判断前后30天是否都在增长
        is_peak = True
        for j in range(1, Days+1):
            if sseci.iloc[i - j] >= sseci.iloc[i] or sseci.iloc[i + j] >= sseci.iloc[i]:
                is_peak = False
                break
        
        if is_peak:
            ls_p[formatted_time] = sseci.iloc[i]  # 将峰值及其索引存储到字典中
        
        # 判断前后30天是否都在下降
        is_trough = True
        for j in range(1, Days+1):
            if sseci.iloc[i - j] <= sseci.iloc[i] or sseci.iloc[i + j] <= sseci.iloc[i]:
                is_trough = False
                break
        
        if is_trough:
            ls_v[formatted_time] = sseci.iloc[i]  # 将谷值及其索引存储到字典中
    
    return ls_p, ls_v

def ASFStock_peak(time_ls, start_date='01/01/2024', end_date='12/31/2024'):
    time = time_ls
    
    # 将日期字符串转换为日期数字
    day = [datetime.strptime(t, '%m/%d/%Y').toordinal() for t in time]
    
    o = []
    n = len(day)
    
    # 计算所有可能的组合
    for b in range(n):
        for c in range(n):
            for d in range(n):
                o.append(day[b] + day[c] - day[d])
    
    o = np.array(o)
    
    # 预测2016年的潜在谷值
    day1 = np.where(o > datetime.strptime(start_date, '%m/%d/%Y').toordinal())[0]
    n1 = o[day1]
    day2 = np.where(n1 < datetime.strptime(end_date, '%m/%d/%Y').toordinal())[0]
    day3 = n1[day2]
    
    # 排序并转换为日期字符串
    m = np.sort(day3)
    k = [datetime.fromordinal(int(date)).strftime('%m/%d/%Y') for date in m]
    
    # 计算频率
    tab = pd.Series(k).value_counts().reset_index()
    tab.columns = ['Date', 'Frequency']
    
    # 使用 t 检验筛选出频率高于均值的日期
    mean_freq = tab['Frequency'].mean()
    std_freq = tab['Frequency'].std()
    t_critical = stats.t.ppf(0.95, len(tab) - 1)
    threshold = mean_freq + t_critical * std_freq
    
    # wsell = tab[tab['Frequency'] > threshold]['Date'].values
    wsell = tab[tab['Frequency'] > 2]['Date'].values
    
    # print('可能的卖出点：')
    # print('月/日/年')
    # print(wsell)
    
    return wsell, tab

def ASFStock_valley(time_ls, start_date='01/01/2024', end_date='12/31/2024'):
    time = time_ls
    
    # 将日期字符串转换为日期数字
    day = [datetime.strptime(t, '%m/%d/%Y').toordinal() for t in time]
    
    o = []
    n = len(day)
    
    # 计算所有可能的组合
    for b in range(n):
        for c in range(n):
            for d in range(n):
                o.append(day[b] + day[c] - day[d])
    
    o = np.array(o)
    
    day1 = np.where(o > datetime.strptime(start_date, '%m/%d/%Y').toordinal())[0]
    n1 = o[day1]
    day2 = np.where(n1 < datetime.strptime(end_date, '%m/%d/%Y').toordinal())[0]
    day3 = n1[day2]
    
    # 排序并转换为日期字符串
    m = np.sort(day3)
    k = [datetime.fromordinal(int(date)).strftime('%m/%d/%Y') for date in m]
    
    # 计算频率
    tab = pd.Series(k).value_counts().reset_index()
    tab.columns = ['Date', 'Frequency']
    
    # 使用 t 检验筛选出频率高于均值的日期
    mean_freq = tab['Frequency'].mean()
    std_freq = tab['Frequency'].std()
    t_critical = stats.t.ppf(0.95, len(tab) - 1)
    threshold = mean_freq + t_critical * std_freq
    
    # wbuy = tab[tab['Frequency'] > threshold]['Date'].values
    wbuy = tab[tab['Frequency'] > 2]['Date'].values
    
    # print('可能的买入点：')
    # print('月/日/年')
    # print(wbuy)
    
    return wbuy, tab

def PlotProcess(year=2024):
    file_path = '上证指数.xlsx'
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    wbuy = pd.read_csv(f'csv/buysignal_{year}.csv')
    wsell = pd.read_csv(f'csv/sellsignal_{year}.csv')
    sseci = pd.read_excel(file_path)
    
    # 确保时间列是 datetime 类型
    sseci['时间'] = pd.to_datetime(sseci['时间'], format='%m/%d/%Y')
    sseci = sseci.set_index('时间')['收盘价(元)']
    
    # 合并买入和卖出信号
    wbuy['type'] = 'buy'
    wsell['type'] = 'sell'
    wbuy['Date'] = pd.to_datetime(wbuy['Date'])
    wsell['Date'] = pd.to_datetime(wsell['Date'])
    
    # 过滤掉不在 sseci 索引中的日期
    wbuy = wbuy[wbuy['Date'].isin(sseci.index)]
    wsell = wsell[wsell['Date'].isin(sseci.index)]
    sseci = sseci.loc[start_date:end_date]
    # print(sseci)
    plt.figure(figsize=(12, 8))
    plt.plot(sseci.loc[start_date:end_date], label='Actual Price')
    plt.scatter(wbuy['Date'], sseci.loc[wbuy['Date']], color='green', marker='v', label='Buy Signal', s=20)
    plt.scatter(wsell['Date'], sseci.loc[wsell['Date']], color='red', marker='^', label='Sell Signal', s=20)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Actual Price')
    plt.tight_layout()
    plt.savefig(f'plot/ASF_predict_{year}.pdf')

def Barplot(tab,name):
    mean_freq = tab['Frequency'].mean()
    std_freq = tab['Frequency'].std()
    t_critical_95 = stats.t.ppf(0.95, len(tab) - 1)
    t_critical_99 = stats.t.ppf(0.99, len(tab) - 1)
    t_critical_90 = stats.t.ppf(0.90, len(tab) - 1)
    t_critical_975 = stats.t.ppf(0.975, len(tab) - 1)
    threshold_90 = mean_freq + t_critical_90 * std_freq
    threshold_99 = mean_freq + t_critical_99 * std_freq
    threshold_975 = mean_freq + t_critical_975 * std_freq
    threshold_95 = mean_freq + t_critical_95 * std_freq
    plt.figure(figsize=(10, 6))
    # plt.hlines(threshold_95, 0, len(tab), colors='r', linestyle='--', label='CI=0.95')  # 虚线
    # plt.hlines(threshold_90, 0, len(tab), colors='b', linestyle='-.', label='CI=0.90')  # 点划线
    # plt.hlines(threshold_99, 0, len(tab), colors='g', linestyle=':', label='CI=0.99')   # 点线
    # plt.hlines(threshold_975, 0, len(tab), colors='m', linestyle='-', label='CI=0.975') # 实线

    tab = tab.sort_values(by='Date')
    plt.bar(tab['Date'], tab['Frequency'])
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(np.arange(0, len(tab['Date']), 50))
    # plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'plot/{name}.pdf')

def find_nearest_trading_date(predicted_dates, trading_dates):
    """
    For each predicted date, find the nearest trading date (before or after).
    
    Args:
        predicted_dates: List or Series of predicted dates (datetime or str)
        trading_dates: Index or list of available trading dates (datetime)
    
    Returns:
        Series of nearest trading dates
    """
    predicted_dates = pd.to_datetime(predicted_dates)
    trading_dates = pd.to_datetime(trading_dates)
    trading_dates = trading_dates[trading_dates >= predicted_dates.min()]
    
    nearest_dates = []
    for date in predicted_dates:
        # Calculate absolute differences (in days)
        abs_diffs =  abs(trading_dates - date)
        # print(abs_diffs)
        abs_diffs=abs_diffs
        # Find the index of the smallest difference
        idx = abs_diffs.argmin()
        nearest_date = trading_dates[idx]
        nearest_dates.append(nearest_date)
    
    return pd.Series(nearest_dates, index=predicted_dates, name='Date')
    
if __name__ =='__main__':
    with open('para.yml') as f:
        settings = yaml.safe_load(f)
    file_path = '上证指数.xlsx'
    year=settings['year']
    start_date = f'01/04/{year}'
    end_date = f'12/30/{year}'
    sseci = pd.read_excel(file_path)
    
    sseci['date'] = pd.to_datetime(sseci['时间'], format='%Y-%m-%d')
    sseci['date']=sseci['date'].dt.strftime('%m/%d/%Y')
    sseci = sseci.set_index('date')['收盘价(元)']
    # ls_p, ls_v = Vs_N_Ps(file_path,Days=40)
    ls_v=['01/02/1992', '12/20/1993', '07/29/1994', '02/07/1995', '01/19/1996', '02/20/1997', '08/18/1998', '05/17/1999', '01/04/2000', '10/22/2001', '01/29/2002', '11/13/2003', '09/13/2004', '06/06/2005', '01/04/2006', '02/06/2007', '10/28/2008', '01/05/2009', '07/02/2010', '12/28/2011', '12/04/2012', '06/25/2013', '03/12/2014', '08/26/2015', '01/27/2016', '05/11/2017', '10/19/2018', '01/04/2019', '03/19/2020', '07/28/2021', '04/27/2022', '12/21/2023']
    ls_p=['05/26/1992', '02/16/1993', '09/13/1994', '05/22/1995', '12/11/1996', '05/12/1997', '06/04/1998', '06/30/1999', '11/23/2000', '06/14/2001', '06/25/2002', '04/16/2003', '04/07/2004', '02/25/2005', '12/29/2006', '10/16/2007', '01/14/2008', '08/04/2009', '01/11/2010', '04/18/2011', '02/27/2012', '02/18/2013', '12/31/2014', '06/12/2015', '01/04/2016', '11/14/2017', '01/29/2018', '04/08/2019', '12/31/2020', '02/18/2021', '01/04/2022', '05/09/2023']
    PoP_v=[]
    PoP_p=[]
    for i in range(len(ls_v)):
        if ls_v[i].split('/')[-1]<str(year):
            PoP_v.append(ls_v[i])

    for i in range(len(ls_p)):
        if ls_p[i].split('/')[-1]<str(year):
            PoP_p.append(ls_p[i])
    ls_v=PoP_v
    ls_p=PoP_p
    # print("峰值：")
    # print(ls_p)
    # print("谷值：")
    # print(ls_v)
    # time_ls = list(ls_p.keys())
    time_ls=ls_p
    wsell, tab = ASFStock_peak(time_ls, start_date, end_date)
    # time_ls = list(ls_v.keys())
    time_ls=ls_v
    wbuy, tab1 = ASFStock_valley(time_ls, start_date, end_date)
    
    wbuy=pd.Series(wbuy, name='Date')
    wsell=pd.Series(wsell, name='Date')

    wbuy = find_nearest_trading_date(wbuy, sseci.index)
    wsell = find_nearest_trading_date(wsell, sseci.index)

    # wbuy = wbuy[wbuy.isin(sseci.index)]
    # wsell = wsell[wsell.isin(sseci.index)]

    wbuy=pd.to_datetime(wbuy).dt.strftime('%m/%d/%Y').sort_values()
    wsell=pd.to_datetime(wsell).dt.strftime('%m/%d/%Y').sort_values()

    wbuy.to_csv(f'csv/buysignal_{year}.csv', index=False)
    wsell.to_csv(f'csv/sellsignal_{year}.csv', index=False)
    PlotProcess(year)
    Barplot(tab,name=f'Sell Signal Frequency_{year}')
    Barplot(tab1,name=f'Buy Signal Frequency_{year}')

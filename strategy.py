import pandas as pd
import matplotlib.pyplot as plt
import yaml
import numpy as np
from Main_Chebyshev5input import WASDSSECI_5inputwnh
global year
with open('para.yml') as f:
    settings = yaml.safe_load(f)
year = settings['year']

def predict():
    filename = '上证指数.xlsx'
    start_date = f'{year}-01-04'  # 统一改为年-月-日格式
    end_date = f'{year}-12-31'    # 修正为有效日期
    rate_test = 0.01
    DF = pd.read_excel(filename)
    
    # 确保时间列解析正确（根据实际Excel中的格式调整）
    DF['时间'] = pd.to_datetime(DF['时间'], format='%Y-%m-%d')  # 如果Excel中是"2024-01-04"格式
    
    # 统一日期解析格式
    start_date = pd.to_datetime(start_date)  # 自动推断或显式指定format='%Y-%m-%d'
    end_date = pd.to_datetime(end_date)
    
    DF['Predict_price'] = DF['收盘价(元)']
    DF = DF.set_index('时间')[['收盘价(元)', 'Predict_price', '开盘价(元)']].loc[start_date:end_date]
    
    # 修正时间格式字符串
    time_ls = DF.index.strftime('%Y-%m-%d').tolist()  # 移除末尾的'Y'
    
    for i in range(len(time_ls)):
        zyntime, Predict_price, entire_t, actual_p, entire_FuncValue = WASDSSECI_5inputwnh(rate_test, time_ls[i])
        DF.loc[time_ls[i], 'Predict_price'] = Predict_price
    
    # 合并信号数据（保持原有逻辑）
    buy_signal = pd.read_csv(f'train/buy_signal_predict_{year}.csv')
    sell_signal = pd.read_csv(f'train/sell_signal_predict_{year}.csv')
    buy_signal['Buy'] = 1
    sell_signal['Sell'] = 1
    
    DF.reset_index(inplace=True)
    DF.rename(columns={'时间': 'Date'}, inplace=True)
    
    # 统一日期类型
    DF['Date'] = pd.to_datetime(DF['Date'])
    buy_signal['Date'] = pd.to_datetime(buy_signal['Date'])
    sell_signal['Date'] = pd.to_datetime(sell_signal['Date'])
    
    DF = DF.merge(buy_signal[['Date', 'Buy']], on='Date', how='outer')
    DF = DF.merge(sell_signal[['Date', 'Sell']], on='Date', how='outer')

    # Check the merged DataFrame
    # print("Columns in merged DF:", DF.columns)
    DF.to_csv(f'train/buy_strategy_{year}.csv')

def pure_buy(Ispure=1, *args):
    cash, buyper, current_open, transaction_cost = args
    if Ispure == 1:
        return (cash * buyper) / (current_open * (1 + transaction_cost))
    elif Ispure == 0:
        return (cash * buyper) // (current_open * (1 + transaction_cost))

def calMDD(df):
    MDD = 0
    peak = -99999
    DD = []
    for i in range(df.shape[0]):
        if df.iloc[i] > peak:
            peak = df.iloc[i]
        DD = np.append(DD, (peak - df.iloc[i])/peak)
        if DD[i] > MDD:
            MDD = DD[i]
    return MDD

def main(DF, transaction_cost=0.001, initial_value=100000000, buyper=0.1, Ispure=1):
    # 初始化
    cash = initial_value
    shares = 0
    portfolio_value = []
    pending_action = None  # 'buy'或'sell'
    Log=[]
    # 获取最后一个卖出信号的索引
    sell_signals = DF[DF['Sell'] == 1]
    last_sell_index = sell_signals.index[-1] if not sell_signals.empty else -1
    Should_sell = 0
    for i in range(0, len(DF)):  # 从1开始避免i-1越界
        current_close = DF.iloc[i]['收盘价(元)']
        current_open = DF.iloc[i]['开盘价(元)']
        prev_close = DF.iloc[i-1]['收盘价(元)']
        current_predict = DF.iloc[i]['Predict_price']
        Date=DF.iloc[i]['Date']
        
        # # 1. 强制平仓条件：最后一个卖出信号无条件执行
        # if i == last_sell_index and shares > 0:
        #     cash += shares * current_close * (1 - transaction_cost)
        #     shares = 0
        #     portfolio_value.append(cash)
        #     pending_action = None
        #     print(f'[强制平仓] 价格:{current_close:.2f} 现金:{cash:.2f}')
        #     continue

        # 3. 捕获新信号（买卖信号互斥）
        if pending_action is None:
            if DF.iloc[i]['Sell'] == 1 and shares > 0:
                pending_action = 'sell'
            elif DF.iloc[i]['Buy'] == 1 and cash > 0:
                if Should_sell == 0:
                    pending_action = 'buy'
            
        # 2. 处理待执行操作
        if pending_action == 'sell':
            if prev_close > current_predict:
                cash += shares * current_close * (1 - transaction_cost)
                shares = 0
                pending_action = None
                Should_sell = 0
                Log.append(['正常卖出', Date, current_open, current_close, cash, shares_to_buy, cost, shares])
                print(f'正常卖出 日期:{Date} 开盘价:{current_open:.2f} 收盘价:{current_close:.2f}  现金:{cash:.2f} 股数:{shares_to_buy} 成本:{cost:.2f} 持仓:{shares}')
                
        elif pending_action == 'buy':
            if prev_close < current_predict:
                shares_to_buy = pure_buy(Ispure, cash, buyper, current_open, transaction_cost)
                cost = shares_to_buy * current_open * (1 + transaction_cost)
                cash -= cost
                shares += shares_to_buy
                pending_action = None
                Should_sell = 1
                Log.append(['正常买入', Date, current_open, current_close, cash,  shares_to_buy, cost, shares])
                print(f'正常买入 日期:{Date} 开盘价:{current_open:.2f} 收盘价:{current_close:.2f}  现金:{cash:.2f} 股数:{shares_to_buy} 成本:{cost:.2f} 持仓:{shares}')
                
        # 4. 记录组合价值
        pv = cash + shares * current_close
        portfolio_value.append(pv)

    pd.DataFrame(Log,columns=['交易类型','日期','开盘价','收盘价','现金','股数','成本','持仓']).to_csv(f'outcome/Log_{year}.csv')
    pd.DataFrame(portfolio_value).to_csv(f'outcome/portfolio_value_{year}.csv')
    pd.DataFrame(DF['收盘价(元)']).to_csv(f'outcome/base_value_{year}.csv')
    final_value = portfolio_value[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    base_return = (DF['收盘价(元)'].iloc[-1] / DF['收盘价(元)'].iloc[0] - 1) * 100
    sharpe_ratio =  (pd.Series(portfolio_value).pct_change() - DF['收盘价(元)'].pct_change()).mean()/ (pd.Series(portfolio_value).pct_change().std())
    MDD = calMDD(pd.Series(portfolio_value))

    base_portfolio_value = [initial_value]
    for i in range(1, len(DF)):
        shares=base_portfolio_value[0]/DF['开盘价(元)'].iloc[0]
        base_portfolio_value.append(DF['收盘价(元)'].iloc[i]*shares)

    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Initial Portfolio Value: {initial_value}")
    print(f"Final Portfolio Value: {final_value}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Base Return: {base_return:.2f}%")
    print(f"Max Drawdown: {MDD:.2f}%")
    Opt=pd.DataFrame({'Initial Portfolio Value': [initial_value], 'Final Portfolio Value': [final_value], 'Total Return': [total_return], 'Base Return': [base_return], 'Sharpe Ratio': [sharpe_ratio],
                      'Max Drawdown': [MDD]}, index=[year])
    Opt.to_csv(f'outcome/strategy_opt_{year}.csv')
    plt.figure(figsize=(12, 8))
    plt.plot(DF['Date'], portfolio_value, label='Portfolio Value')
    plt.plot(DF['Date'], base_portfolio_value, label='Base Portfolio Value')
    plt.xlabel('Date')
    plt.xticks(DF['Date'][::50])
    plt.ylabel('Portfolio Value')
    # plt.title('Portfolio Value Over Time')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'portfolio_value_{year}.pdf')

def draw(Base_PV,Total_PV,Total_PV_noTC):
    Base_PV=Base_PV*10**8/Base_PV.iloc[0]
    plt.figure(figsize=(12, 8))
    print(Total_PV)
    plt.plot(Total_PV, label='With 1‰ Transaction Cost')
    plt.plot(Base_PV, label='Base Portfolio Value')
    print(Total_PV_noTC)
    plt.plot(Total_PV_noTC, label='Without Transaction Cost')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    # plt.xticks(DF['Date'][::50])
    plt.legend()
    plt.tight_layout()
    plt.savefig('portfolio_value.pdf')

if __name__ == '__main__':
    # predict()
    for year in range(2016,2025,1):
        DF=pd.read_csv(f'train/buy_strategy_{year}.csv')
        main(DF,transaction_cost = 0.001,initial_value=100000000,buyper=1)
    # DF=pd.read_csv(f'train/buy_strategy_{year}.csv')
    # main(DF,transaction_cost = 0,initial_value=100000000,buyper=1,Ispure=0)
    DF=pd.DataFrame()
    for year in range(2016,2025,1):
        df=pd.read_csv(f'outcome/strategy_opt_{year}.csv')
        DF=pd.concat([DF,df])
    
    Total_PV=pd.read_csv(f'outcome/portfolio_value_{2016}.csv',index_col=0)
    Base_PV=pd.read_csv(f'outcome/base_value_{2016}.csv',index_col=0)
    for year in range(2017,2025,1):
        pv=pd.read_csv(f'outcome/portfolio_value_{year}.csv',index_col=0)
        bpv=pd.read_csv(f'outcome/base_value_{year}.csv',index_col=0)
        pv=Total_PV.iloc[-1]/pv.iloc[0]*pv
        bpv=Base_PV.iloc[-1]/bpv.iloc[0]*bpv
        Total_PV=pd.concat([Total_PV,pv])
        Base_PV=pd.concat([Base_PV,bpv])
    Total_PV.index=range(0,len(Total_PV))
    # Total_PV.to_csv('outcome/portfolio_value.csv')
    Base_PV.index=range(0,len(Base_PV))
    # Base_PV.to_csv('outcome/base_value.csv')
    TR=Total_PV.pct_change().dropna().rename(columns={'0':'R'})
    BR=Base_PV.pct_change().dropna().rename(columns={'收盘价(元)':'R'})
    
    total_return = Total_PV.iloc[-1]/Total_PV.iloc[0]*100-100
    base_return = (Base_PV.iloc[-1]/Base_PV.iloc[0] - 1) * 100
    sharpe_ratio = (TR - BR).mean() / (TR.std())
    MDD = calMDD(Total_PV.squeeze())
    total=pd.DataFrame({'Initial Portfolio Value': [DF['Initial Portfolio Value'].iloc[0]], 'Final Portfolio Value': [DF['Final Portfolio Value'].iloc[-1]], 'Total Return': [total_return.iloc[0]], 'Base Return': [base_return.iloc[0]], 'Sharpe Ratio': [sharpe_ratio.iloc[0]],
                      'Max Drawdown': [MDD]}, index=['Total'])
    DF=pd.concat([DF,total])
    DF.to_csv('outcome/strategy_opt.csv')
    Base_PV.to_csv('draw/base_value.csv')
    Total_PV.to_csv('draw/portfolio_value.csv')

    Base_PV=pd.read_csv('draw/base_value.csv',index_col=0)
    Total_PV=pd.read_csv('draw/portfolio_value.csv',index_col=0)
    Total_PV_noTC=pd.read_csv('draw/portfolio_value_noTC.csv',index_col=0)
    draw(Base_PV,Total_PV,Total_PV_noTC)


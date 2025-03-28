import numpy as np
import pandas as pd
from datetime import datetime
import time
from scipy.linalg import pinv
from numpy.linalg import norm
import yaml

def WASDSSECI_5inputwnh(rate_best: int, date_pre: int, Traindays: int = 60, InputVars: int = 5):
    '''
        Perform prediction using a 5 input Chebyshev polynomial-based neural network model.

            rate_best (int): The best rate parameter for normalization.
            date_pre (int): The Date for which the prediction is to be made.
            Traindays (int, optional): The number of training days. Defaults to 60.
            InputVars (int, optional): The number of input variables. Defaults to 5.

            tuple: A tuple containing:
                - zyntime (np.ndarray): The input data for the prediction Date.
                - Predict_price (float): The predicted price for the given Date.
                - entire_t (np.ndarray): The entire training data used.
                - actual_p (float): The actual price on the prediction Date.
                - entire_FuncValue (np.ndarray): The function values for the entire training data.
    '''
    # 读取Excel数据
    filename = '上证指数.xlsx'
    data = pd.read_excel(filename, header=0)
    data = data.values  
    # 处理日期列
    date_col = data[:, 0]
    converted_dates = []
    for d in date_col:
        d=d.toordinal()
        converted_dates.append(d)
    data[:, 0] = np.array(converted_dates)
    data.astype(np.float64)
    # 查找预测日期位置
    date_pre = pd.to_datetime(date_pre).toordinal()
    date_prenum = np.where(data[:, 0] == date_pre)[0][0]
    # 提取数据段
    ydata = data[date_prenum-Traindays:date_prenum, 3:InputVars+1].astype(np.float64)
    entire_t = np.vstack((data[date_prenum-Traindays:date_prenum, 0:2].T, ydata.T)).astype(np.float64)
    entire_y = data[date_prenum-Traindays:date_prenum, 2].T.astype(np.float64)
    actual_p = data[date_prenum, 2]
    rate = rate_best
    length_t = entire_t.shape[1]
    # **正确的计算方式** 这个归一化是决定神经网络最好性能的关键
    entire_t_fortran = entire_t.flatten('F')
    cal_t = (entire_t - entire_t_fortran[0])* (rate + 1)  / (entire_t_fortran[2] - entire_t_fortran[0])-1
    u = cal_t
    # np.save('u.npy', u)
    targetFuncValue = entire_y
    entire_FuncValue = entire_y.copy()
    lengthU = length_t
    PolynomialsValueu_Chebyshev = np.zeros((u.shape[1], 2*InputVars))
    PolynomialsValueu_Chebyshev[:, 0:InputVars] = (1+0*u).T
    PolynomialsValueu_Chebyshev[:, InputVars:2*InputVars] = u.T
    currentHidNeuNum = 2
    MinError = np.inf
    MinErr_HidNeuNum = 2
    CurrentResult = 1000
    start_time = time.time()
    currentMSE=np.array([])
    # Training loop
    while CurrentResult <= MinError or currentHidNeuNum < MinErr_HidNeuNum + 2:
        currentHidNeuNum += 1
        cols=2 * u.T * PolynomialsValueu_Chebyshev[:, (currentHidNeuNum - 1) * InputVars - InputVars : (currentHidNeuNum - 1) * InputVars] - PolynomialsValueu_Chebyshev[:, (currentHidNeuNum - 2) * InputVars - InputVars : (currentHidNeuNum - 2) * InputVars]
        PolynomialsValueu_Chebyshev =np.hstack((PolynomialsValueu_Chebyshev, cols)) 
        # print(targetFuncValue)
        Weights = pinv(PolynomialsValueu_Chebyshev) @ targetFuncValue.T
        # Compute the output and error
        netOutputTemp = PolynomialsValueu_Chebyshev @ Weights
        error = targetFuncValue.T - netOutputTemp
        # print('error:', error)
        # np.save('PolynomialsValueu_Chebyshev.npy', PolynomialsValueu_Chebyshev.T)
        squared_error = error**2 
        currentMSE = np.hstack([currentMSE,np.sum(squared_error) / lengthU])
        CurrentResult=currentMSE[currentHidNeuNum-3]
        if CurrentResult < MinError:
            MinError = CurrentResult
            MinErr_HidNeuNum = currentHidNeuNum
            netOutput = netOutputTemp.T
    print(PolynomialsValueu_Chebyshev.shape,Weights.shape,targetFuncValue.shape)
    # 预测部分
    zyntime = np.hstack((data[date_prenum, 0:2], data[date_prenum, 3:InputVars+1]))
    # **正确的计算方式** 这个归一化是决定神经网络最好性能的关键
    u_predict = (zyntime - entire_t_fortran[0])* (rate + 1)  / (entire_t_fortran[2] - entire_t_fortran[0])-1 
    Weights_MinErr = pinv(PolynomialsValueu_Chebyshev[:, : MinErr_HidNeuNum*InputVars]) @ targetFuncValue.T
    # 预测多项式生成
    PolynomialsValueu_Chebyshev_predict = np.zeros((u_predict.shape[0], 2*InputVars))
    PolynomialsValueu_Chebyshev_predict[:, :InputVars] = (1+0*u_predict).T
    PolynomialsValueu_Chebyshev_predict[:, InputVars:2*InputVars] = u_predict.T
    for currentHidNeuNum_predict in range(3, MinErr_HidNeuNum+1):
        # print(u_predict.shape,PolynomialsValueu_Chebyshev_predict.shape)
        new_cols = 2 * u_predict.T * PolynomialsValueu_Chebyshev_predict[:, (currentHidNeuNum_predict-1)*InputVars-InputVars:(currentHidNeuNum_predict-1)*InputVars]- PolynomialsValueu_Chebyshev_predict[:, (currentHidNeuNum_predict-2)*InputVars-InputVars:(currentHidNeuNum_predict-2)*InputVars]
        PolynomialsValueu_Chebyshev_predict = np.hstack((PolynomialsValueu_Chebyshev_predict,new_cols))
    # 计算预测结果
    PolynomialsValueu_Chebyshev_predict = PolynomialsValueu_Chebyshev_predict.astype(np.float64)
    netOutputTemp_predict = PolynomialsValueu_Chebyshev_predict[:, :MinErr_HidNeuNum*InputVars] @ Weights_MinErr
    Predict_price = netOutputTemp_predict.T[0]
    # print(PolynomialsValueu_Chebyshev_predict.shape, Weights_MinErr.shape)
    return zyntime, Predict_price, entire_t, actual_p, entire_FuncValue

if __name__ == '__main__':
    with open('para.yml') as f:
        settings = yaml.safe_load(f)
    rate_test = 0.01
    file_path = '上证指数.xlsx'
    year=settings['year']
    # start_date = f'01/03/{year}'
    # end_date = f'12/31/{year}'
    buy_signal = pd.read_csv(f'csv/buysignal_{year}.csv')
    sell_signal = pd.read_csv(f'csv/sellsignal_{year}.csv')
    sseci = pd.read_excel(file_path)
    sseci['date'] = pd.to_datetime(sseci['时间'], format='%Y-%m-%d')
    sseci['date']=sseci['date'].dt.strftime('%m/%d/%Y')
    sseci['year'] = sseci['时间'].dt.year
    sseci = sseci[sseci['year'] == year]
    sseci = sseci.set_index('date')['收盘价(元)']
    buy_signal = buy_signal[buy_signal['Date'].isin(sseci.index)].dropna()
    sell_signal = sell_signal[sell_signal['Date'].isin(sseci.index)].dropna()
    print(buy_signal)
    results = []
    results1 = []
    for i in range(len(buy_signal)):
        print(buy_signal['Date'].iloc[i])
        date_pre = buy_signal['Date'].iloc[i] 
        zyntime, Predict_price, entire_t, actual_p, entire_FuncValue = WASDSSECI_5inputwnh(rate_test, date_pre)
        print(Predict_price, actual_p)
        results.append([buy_signal['Date'].iloc[i], Predict_price, actual_p,  entire_t[1,-1]])
    
    # 创建 DataFrame 并保存到 CSV 文件
    DF = pd.DataFrame(results, columns=['Date', 'Predict_price', 'actual_p','actual_p-1'])
    DF= DF.loc[DF['Predict_price'] >0]
    DF.to_csv(f'train/buy_signal_predict_{year}.csv', index=False)
    
    for i in range(len(sell_signal)):
        print(sell_signal['Date'].iloc[i])
        date_pre = sell_signal['Date'].iloc[i] 
        zyntime, Predict_price, entire_t, actual_p, entire_FuncValue = WASDSSECI_5inputwnh(rate_test, date_pre)
        print(Predict_price, actual_p)
        results1.append([sell_signal['Date'].iloc[i], Predict_price, actual_p, entire_t[1,-1]])

    # 创建 DataFrame 并保存到 CSV 文件
    DF1 = pd.DataFrame(results1, columns=['Date', 'Predict_price', 'actual_p','actual_p-1'])
    DF1= DF1.loc[DF1['Predict_price'] >0]
    DF1.to_csv(f'train/sell_signal_predict_{year}.csv', index=False)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
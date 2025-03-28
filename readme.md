中文版


这个项目主要涉及使用 Chebyshev 多项式和神经网络模型对上证指数（SSECI）进行预测，并基于预测结果生成买卖信号。项目包含多个 Python 脚本，每个脚本执行特定的任务。以下是对项目中各个文件的介绍：

### 1. Main_Chebyshev5input.py
这个脚本实现了一个基于 Chebyshev 多项式的神经网络模型，用于预测上证指数的价格。主要功能包括：
- 读取 Excel 文件中的数据。
- 处理日期列并进行归一化。
- 使用 Chebyshev 多项式生成特征矩阵。
- 训练神经网络模型并进行预测。
- 将预测结果保存到 CSV 文件中。

### 2. Plot_Chebyshev.py
这个脚本用于绘制上证指数的实际价格和预测价格，并标注买卖信号。主要功能包括：
- 读取实际价格和买卖信号的 CSV 文件。
- 计算预测价格与实际价格的差值。
- 绘制实际价格曲线和买卖信号的柱状图。
- 将绘制的图表保存为 PDF 文件。

### 3. strategy.py
这个脚本实现了一个简单的交易策略，基于预测的买卖信号进行交易模拟。主要功能包括：
- 读取实际价格和买卖信号的 CSV 文件。
- 根据买卖信号进行交易操作，计算投资组合的价值。
- 绘制投资组合价值随时间变化的曲线，并保存为 PDF 文件。

### 4. ASF_predict.py
这个脚本用于识别上证指数的局部峰值和谷值，并基于这些点生成买卖信号。主要功能包括：
- 读取 Excel 文件中的数据。
- 识别局部峰值和谷值。
- 使用 t 检验计算显著性阈值，筛选出显著的峰值和谷值。 CI=95%
- 将买卖信号保存到 CSV 文件中，并绘制图表。

### 5. Verification_Chebyshev5input.py
这个脚本用于验证 Chebyshev 多项式模型的预测误差。主要功能包括：
- 读取预测结果的 CSV 文件。
- 计算预测价格与实际价格的相对误差。
- 输出平均相对误差。

### 6. train 目录
这个目录包含买卖信号的预测结果 CSV 文件，如 buy_signal_predict.csv 和 sell_signal_predict.csv。

### 7. plot 目录
这个目录包含生成的图表 PDF 文件，如 `ASF_predict_CHEBYSHEV.pdf` 和 `Sensitivity.pdf`。

### 8. csv 目录
这个目录包含原始买卖信号的 CSV 文件，如 buysignal.csv 和 sellsignal.csv。通过这些脚本和文件，项目实现了对上证指数的预测、买卖信号的生成和交易策略的模拟，并提供了相应的可视化图表。

运行顺序 ASF_predict.py ---- Main_Chebyshev5input --- Plot_Chebyshev --- Verification_Chebyshev --- strategy


English Version

Here's the English translation of your project description:

This project primarily involves using Chebyshev polynomials and neural network models to predict the Shanghai Stock Exchange Composite Index (SSECI) and generate buy/sell signals based on the predictions. The project consists of multiple Python scripts, each performing specific tasks. Below is an introduction to each file in the project:

### 1. Main_Chebyshev5input.py  
This script implements a neural network model based on Chebyshev polynomials to predict the SSECI price. Its main functions include:  
- Reading data from an Excel file.  
- Processing the date column and performing normalization.  
- Generating a feature matrix using Chebyshev polynomials.  
- Training the neural network model and making predictions.  
- Saving the prediction results to a CSV file.  

### 2. Plot_Chebyshev.py  
This script is used to plot the actual and predicted prices of the SSECI and annotate buy/sell signals. Its main functions include:  
- Reading CSV files containing actual prices and buy/sell signals.  
- Calculating the difference between predicted and actual prices.  
- Plotting the actual price curve and bar charts for buy/sell signals.  
- Saving the generated plots as PDF files.  

### 3. strategy.py  
This script implements a simple trading strategy to simulate transactions based on predicted buy/sell signals. Its main functions include:  
- Reading CSV files containing actual prices and buy/sell signals.  
- Executing trades based on buy/sell signals and calculating portfolio value.  
- Plotting the portfolio value over time and saving the chart as a PDF file.  

### 4. ASF_predict.py  
This script identifies local peaks and troughs in the SSECI and generates buy/sell signals based on these points. Its main functions include:  
- Reading data from an Excel file.  
- Identifying local peaks and troughs.  
- Using a t-test to calculate significance thresholds (95% CI) and filtering significant peaks and troughs.  
- Saving buy/sell signals to a CSV file and generating plots.  

### 5. Verification_Chebyshev5input.py  
This script validates the prediction errors of the Chebyshev polynomial model. Its main functions include:  
- Reading prediction results from a CSV file.  
- Calculating the relative error between predicted and actual prices.  
- Outputting the average relative error.  

### 6. train Directory  
This directory contains CSV files with buy/sell signal prediction results, such as `buy_signal_predict.csv` and `sell_signal_predict.csv`.  

### 7. plot Directory  
This directory contains generated PDF charts, such as `ASF_predict_CHEBYSHEV.pdf` and `Sensitivity.pdf`.  

### 8. csv Directory  
This directory contains original buy/sell signal CSV files, such as `buysignal.csv` and `sellsignal.csv`.  

Through these scripts and files, the project achieves SSECI prediction, buy/sell signal generation, and trading strategy simulation, along with corresponding visualizations.  

**Execution Order**:  
`ASF_predict.py` → `Main_Chebyshev5input.py` → `Plot_Chebyshev.py` → `Verification_Chebyshev.py` → `strategy.py`  

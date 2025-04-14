import math
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import linregress
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Suppress warnings and set plot style
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

class StockAnalyzer:
    def __init__(self, data_path):
        """Initialize the StockAnalyzer with data from the specified path."""
        self.df = pd.read_excel(data_path)
        self.df = self.df.sort_values(by=['Time'])
        self.df = self.df.set_index(pd.DatetimeIndex(self.df['Time'].values))
        
    def plot_price_history(self):
        """Plot the historical price data."""
        plt.figure(figsize=(16,8))
        plt.plot(self.df['Target'], label='Final Price')
        plt.title('Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()
        
    def plot_candlestick(self):
        """Create a candlestick chart of the stock."""
        fig = go.Figure(data=[go.Candlestick(
            x=self.df.index,
            low=self.df['Low'],
            high=self.df['High'],
            close=self.df['Target'],
            open=self.df['Open'],
            increasing_line_color='green',
            decreasing_line_color='red')
        ])
        fig.update_layout(title='Candlestick Chart', 
                         yaxis_title='Volume', 
                         xaxis_title='Date')
        fig.show()
        
    def calculate_trend_lines(self, days=240):
        """Calculate and plot trend lines for the specified number of days."""
        df2 = self.df.tail(days)
        data0 = df2.copy()
        data0['date_id'] = ((data0.index.date - data0.index.date.min())).astype('timedelta64[D]')
        data0['date_id'] = data0['date_id'].dt.days + 1
        
        # Calculate high trend
        data1 = data0.copy()
        while len(data1) > 3:
            reg = linregress(x=data1['date_id'], y=data1['High'])
            data1 = data1.loc[data1['High'] > reg[0] * data1['date_id'] + reg[1]]
        reg = linregress(x=data1['date_id'], y=data1['High'])
        data0['high_trend'] = reg[0] * data0['date_id'] + reg[1]
        
        # Calculate low trend
        data1 = data0.copy()
        while len(data1) > 3:
            reg = linregress(x=data1['date_id'], y=data1['Low'])
            data1 = data1.loc[data1['Low'] < reg[0] * data1['date_id'] + reg[1]]
        reg = linregress(x=data1['date_id'], y=data1['Low'])
        data0['low_trend'] = reg[0] * data0['date_id'] + reg[1]
        
        # Plot trends
        plt.figure(figsize=(16,8))
        data0['Target'].plot()
        data0['high_trend'].plot()
        data0['low_trend'].plot()
        plt.show()
        
    def calculate_macd(self, days=240):
        """Calculate and plot MACD indicator."""
        df2 = self.df.tail(days)
        shortEMA = df2.Target.ewm(span=12, adjust=False).mean()
        longEMA = df2.Target.ewm(span=26, adjust=False).mean()
        MACD = shortEMA - longEMA
        signal = MACD.ewm(span=9, adjust=False).mean()
        
        df2['MACD'] = MACD
        df2['signal line'] = signal
        
        plt.figure(figsize=(16,8))
        plt.plot(df2.index, MACD, label='MACD', color='red', alpha=0.7)
        plt.plot(df2.index, signal, label='Signal', color='blue', alpha=0.7)
        plt.title('MACD INDICATOR')
        plt.xlabel('Date')
        plt.ylabel('INDICATOR')
        plt.legend()
        plt.show()
        
        return df2
        
    def calculate_moving_averages(self):
        """Calculate and plot moving averages."""
        ma30 = pd.DataFrame()
        ma30['AM'] = self.df['Target'].rolling(window=30).mean()
        ma90 = pd.DataFrame()
        ma90['AM'] = self.df['Target'].rolling(window=90).mean()
        
        plt.figure(figsize=(16,8))
        plt.plot(self.df['Target'], label='Stock')
        plt.plot(ma30, label='MA30')
        plt.plot(ma90, label="MA90")
        plt.title('Stock Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('PRICE')
        plt.legend()
        plt.show()
        
    def predict_with_svm_lr(self, forecast_days=7):
        """Predict stock prices using SVM and Linear Regression."""
        df_target = self.df[['Target']]
        df_target['Prediction'] = df_target[['Target']].shift(-forecast_days)
        
        x = df_target[['Target']][:-forecast_days]
        y = df_target['Prediction'][:-forecast_days]
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
        
        # SVM Model
        mysvr = SVR(kernel='rbf', C=1000, gamma=0.2)
        mysvr.fit(xtrain, ytrain)
        svm_conf = mysvr.score(xtrain, ytrain)
        print('SVM confidence:', svm_conf)
        
        # Linear Regression Model
        lr = LinearRegression()
        lr.fit(xtrain, ytrain)
        lr_conf = lr.score(xtest, ytest)
        print('LR confidence:', lr_conf)
        
        # Make predictions
        x_forecast = np.array(self.df[['Target']])[-forecast_days:]
        lr_pred = lr.predict(x_forecast)
        svm_pred = mysvr.predict(x_forecast)
        
        return lr_pred, svm_pred
        
    def predict_with_lstm(self, lookback_days=7, forecast_days=1):
        """Predict stock prices using LSTM model."""
        data = self.df.filter(['Target'])
        dataset = data.values
        training_data_len = math.ceil(len(dataset) * 0.8)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Prepare training data
        training_data = scaled_data[0:training_data_len, :]
        xtrain, ytrain = [], []
        
        for i in range(lookback_days, len(training_data)):
            xtrain.append(training_data[i-lookback_days:i, 0])
            ytrain.append(training_data[i, 0])
            
        xtrain, ytrain = np.array(xtrain), np.array(ytrain)
        xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
        
        # Build and train LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(lookback_days,1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(xtrain, ytrain, epochs=10, batch_size=1)
        
        # Make predictions
        test_data = scaled_data[training_data_len - lookback_days:, :]
        xtest = []
        ytest = dataset[training_data_len:, :]
        
        for i in range(lookback_days, len(test_data)):
            xtest.append(test_data[i-lookback_days:i, 0])
            
        xtest = np.array(xtest)
        xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
        
        predictions = model.predict(xtest)
        predictions = scaler.inverse_transform(predictions)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(np.square(predictions - ytest)))
        print('RMSE:', rmse)
        
        return predictions

def main():
    """Main function to demonstrate the StockAnalyzer class."""
    analyzer = StockAnalyzer('RawData21.xlsx')
    
    # Visualize data
    analyzer.plot_price_history()
    analyzer.plot_candlestick()
    analyzer.calculate_trend_lines()
    analyzer.calculate_macd()
    analyzer.calculate_moving_averages()
    
    # Make predictions
    lr_pred, svm_pred = analyzer.predict_with_svm_lr()
    lstm_pred = analyzer.predict_with_lstm()
    
    print("Linear Regression Predictions:", lr_pred)
    print("SVM Predictions:", svm_pred)
    print("LSTM Predictions:", lstm_pred)

if __name__ == "__main__":
    main()
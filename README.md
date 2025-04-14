# Stock Price Analysis and Prediction

This project provides tools for analyzing and predicting stock prices using various methods including technical analysis and machine learning models.

## Features

- Price history visualization
- Candlestick chart plotting
- Trend line analysis
- MACD indicator calculation
- Moving averages analysis
- Price prediction using:
  - Support Vector Machine (SVM)
  - Linear Regression
  - Long Short-Term Memory (LSTM) neural network

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your stock data in an Excel file named `RawData21.xlsx` with the following columns:
   - Time
   - Open
   - High
   - Low
   - Target (Closing price)

2. Run the main script:
```bash
python stock_analyzer.py
```

## Example

```python
from stock_analyzer import StockAnalyzer

# Initialize analyzer with your data
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
```

## Project Structure

- `stock_analyzer.py`: Main class containing all analysis and prediction methods
- `requirements.txt`: List of required Python packages
- `RawData21.xlsx`: Input data file (to be provided by user)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
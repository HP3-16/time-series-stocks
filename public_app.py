import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from additional import DatasetGenerator
import keras
import requests
from io import StringIO
import pandas_ta as ta
import plotly.graph_objects as go
import datetime as dt
import yfinance as yf
from plotly.subplots import make_subplots
from datetime import timedelta
from datetime import date

st.title("Stock Prediction")

user_input = st.text_input("Enter the Stock Symbol","RELIANCE")
url = f'https://raw.githubusercontent.com/HP3-16/time-series-stocks/master/equities/{user_input}.csv'
response = requests.get(url)
RELIANCE = pd.read_csv(StringIO(response.text))


st.subheader('Descriptive Data')
st.write(RELIANCE.describe())

FILTERED_RELIANCE = RELIANCE.loc[RELIANCE['Date'] > '2020-01-01']                              
infoType = st.radio("Indicators: ",('Moving Average Chart', 'Trend','Stochastic Oscillator', 'RSI' ))

if infoType == 'Moving Average Chart':
    st.subheader('Closing Price vs Time Chart with 100 MA')
    ma100 = FILTERED_RELIANCE.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12, 6))
    plt.plot(FILTERED_RELIANCE['Date'][FILTERED_RELIANCE['Date'] > '2018-01-01'],ma100,label="100 MA")
    plt.plot(FILTERED_RELIANCE['Date'][FILTERED_RELIANCE['Date'] > '2018-01-01'],FILTERED_RELIANCE.Close)
    # st.pyplot(fig)    
    # st.subheader('Closing Price vs Time Chart with 100 MA & 200MA')
    ma100 = FILTERED_RELIANCE.Close.rolling(100).mean()
    ma200 = FILTERED_RELIANCE.Close.rolling(200).mean()
    fig = plt.figure(figsize = (12, 6))
    plt.plot(FILTERED_RELIANCE['Date'][FILTERED_RELIANCE['Date'] > '2018-01-01'],ma100, 'g',label="100 MA")
    plt.plot(FILTERED_RELIANCE['Date'][FILTERED_RELIANCE['Date'] > '2018-01-01'],ma200, 'r',label ="200 MA")
    plt.plot(FILTERED_RELIANCE['Date'][FILTERED_RELIANCE['Date'] > '2018-01-01'],FILTERED_RELIANCE.Close, 'b')
    plt.legend()
    st.pyplot(fig)

elif infoType == 'Trend':
    start = dt.datetime.today() - dt.timedelta(2 * 365)
    end = dt.datetime.today()
    fig = go.Figure(
        data=go.Scatter(x=FILTERED_RELIANCE.Date, y=FILTERED_RELIANCE['Adj Close'])
    )
    fig.update_layout(
        title={
    'text': "Stock Prices Over Past Few Years",
    'y': 0.9,
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)
    
elif infoType == 'Stochastic Oscillator':
    ShortEMA = FILTERED_RELIANCE["Close"].ewm(span = 12 , adjust = False).mean()
    LongEMA = FILTERED_RELIANCE["Close"].ewm(span = 26, adjust = False).mean()
    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span = 9, adjust = False).mean()

    FILTERED_RELIANCE["MACD"] = MACD
    FILTERED_RELIANCE["Signal Line"] = signal    

    FILTERED_RELIANCE['14-Low'] = FILTERED_RELIANCE['Low'].rolling(14).min()
    FILTERED_RELIANCE['14-High'] = FILTERED_RELIANCE['High'].rolling(14).max()    

    FILTERED_RELIANCE['%K'] = (FILTERED_RELIANCE['Close'] -FILTERED_RELIANCE['14-Low'] )*100/(FILTERED_RELIANCE['14-High'] -FILTERED_RELIANCE['14-Low'] )
    FILTERED_RELIANCE['%D'] = FILTERED_RELIANCE['%K'].rolling(3).mean()    

    fig2 = plt.figure(figsize = (16, 10))
    fig2 = make_subplots(rows=2, cols=1)

    fig2.append_trace(go.Scatter(x=FILTERED_RELIANCE.index, y=FILTERED_RELIANCE['Open'], line=dict(color='#ff9900', width=1),name='Open',legendgroup='1',), row=1, col=1 )    
    # Candlestick
    fig2.append_trace( go.Candlestick(x=FILTERED_RELIANCE.index, open=FILTERED_RELIANCE['Open'], high=FILTERED_RELIANCE['High'], low=FILTERED_RELIANCE['Low'], 
            close=FILTERED_RELIANCE['Close'], increasing_line_color='#ff9900', decreasing_line_color='black', 
                                    showlegend=False ), row=1, col=1)    
    # Fast Signal (%k)
    fig2.append_trace(go.Scatter(x=FILTERED_RELIANCE.index, y=FILTERED_RELIANCE['%K'], line=dict(color='#ff9900', width=2), name='macd',
            # showlegend=False,
            legendgroup='2',), row=2, col=1)    
    # SLow signal (%d)
    fig2.append_trace(go.Scatter(x=FILTERED_RELIANCE.index, y=FILTERED_RELIANCE['%D'], line=dict(color='#000000', width=2),
            # showlegend=False,
            legendgroup='2', name='signal'), row=2, col=1)    
    # Colorize the histogram values
    colors = np.where(FILTERED_RELIANCE['MACD'] < 0, '#000', '#ff9900')
    # Plot the histogram
    fig2.append_trace(go.Bar(x=FILTERED_RELIANCE.index, y=FILTERED_RELIANCE['MACD'], name='histogram', marker_color=colors, ), row=2, col=1)    
    # Make it pretty
    layout = go.Layout(autosize=False,
        width=1000,
        height=1000, plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',font_color='#000000', font_size=20,
        xaxis=dict(
            rangeslider=dict(visible=True) ))    
    fig2.update_layout(layout)
    st.plotly_chart(fig2)

else:
    FILTERED_RELIANCE["RSI(2)"]= ta.rsi(FILTERED_RELIANCE['Close'], length= 2)
    FILTERED_RELIANCE["RSI(7)"]= ta.rsi(FILTERED_RELIANCE['Close'], length= 7)
    FILTERED_RELIANCE["RSI(14)"]= ta.rsi(FILTERED_RELIANCE['Close'], length= 14)
    FILTERED_RELIANCE["CCI(30)"]= ta.cci(close=FILTERED_RELIANCE['Close'],length=30, high= FILTERED_RELIANCE["High"], low =  FILTERED_RELIANCE["Low"])
    FILTERED_RELIANCE["CCI(50)"]= ta.cci(close= FILTERED_RELIANCE['Close'],length= 50, high= FILTERED_RELIANCE["High"], low =  FILTERED_RELIANCE["Low"])
    FILTERED_RELIANCE["CCI(100)"]= ta.cci(close= FILTERED_RELIANCE['Close'],length= 100, high= FILTERED_RELIANCE["High"], low =  FILTERED_RELIANCE["Low"])    
    fig3= plt.figure(figsize=(15,15))
    ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
    ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
    ax1.plot(FILTERED_RELIANCE.Date,FILTERED_RELIANCE['Close'], linewidth = 1.5)
    ax1.set_title('Close PRICE')
    ax2.plot(FILTERED_RELIANCE['RSI(14)'], color = 'orange', linewidth = 1.0)
    ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
    ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
    ax2.set_title('RELATIVE STRENGTH INDEX')
    st.pyplot(fig3)

press_here = st.button("Predict")
if press_here:
    st.subheader("Prediction of Stocks")
    model = keras.models.load_model('modelkeras/saved_model.keras')
    ed_date = '2023-06-01'
    RELIANCE["Date"] = pd.to_datetime(RELIANCE["Date"])
    RELIANCE_Train_X, RELIANCE_Train_Y, RELIANCE_Test_X, RELIANCE_Test_Y,scale = DatasetGenerator.Dataset(RELIANCE, ed_date)
    
    y_predicted = model.predict(RELIANCE_Test_X)
    
    y_predicted = y_predicted/scale
    RELIANCE_Test_X = RELIANCE_Test_X/scale
    split_date = '2023'
    fig = plt.figure(figsize=(20,12))
    plt.plot(RELIANCE['Date'][RELIANCE['Date'] < '2023-06-01'], RELIANCE['Adj Close'][RELIANCE['Date'] < '2023-06-01'], label = 'Training')
    plt.plot(RELIANCE['Date'][RELIANCE['Date'] >= '2023-06-01'], RELIANCE['Adj Close'][RELIANCE['Date'] >= '2023-06-01'], label = 'Testing')
    plt.plot(RELIANCE['Date'][RELIANCE['Date'] >= '2023-06-14'], y_predicted.reshape(-1), label = 'Predictions')
    plt.xlim(pd.Timestamp("2020-01-01"),pd.Timestamp("2024-04-30"))
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend(loc = 'best')
    st.pyplot(fig)
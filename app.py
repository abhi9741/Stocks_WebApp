import prophet
import yfinance
import streamlit
from datetime import date 
from plotly import graph_objs 
from prophet.plot import plot_plotly

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

streamlit.title("Stock Market Forecasting")

# stocks = ("AAPL","GOOG","MSFT","AMZN") #Change this to use input and add other stock options
# selected_stock = streamlit.selectbox("Select a stock from below for forecasting",stocks) #add a input text widget
selected_stock = streamlit.text_input("Type a stock name of choice for forecasting. eg: AAPL, GOOG, MSFT, AMZN r.t.c ", value="AAPL")


@streamlit.cache_data #cache the data instead of relading everytime
def load_data(stock):
    """Loading the stock data"""
    data = yfinance.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = streamlit.text("Load data.....")
data = load_data(selected_stock)
data_load_state.text(f"Loading data {selected_stock} stock data...done!")

#first look at the data
streamlit.subheader('An initial look at the data')
streamlit.write(data.tail())

# add some simple stock analysis here 

def plot_stock_open_close():
    fig = graph_objs.Figure()
    fig.add_trace(graph_objs.Scatter(x=data["Date"], y=data["Open"], name ="Opening Price",line=dict(color="#FA5632")))
    fig.add_trace(graph_objs.Scatter(x=data["Date"], y=data["Close"], name ="Closing Price",line=dict(color="#32AEFA")))
    fig.layout.update(title_text="Opening and Closing prices ", xaxis_rangeslider_visible=True)
    streamlit.plotly_chart(fig)
plot_stock_open_close()

def plot_stock_vol():
    fig = graph_objs.Figure()
    fig.add_trace(graph_objs.Scatter(x=data["Date"], y=data["Volume"], name ="Trading Volume",line=dict(color="#32AEFA")))
    # fig.add_trace(graph_objs.Scatter(x=data["Date"], y=data["Close"], name ="Closing Price",line=dict(color="#32AEFA")))
    fig.layout.update(title_text="Trading Volume", xaxis_rangeslider_visible=True)
    streamlit.plotly_chart(fig)
plot_stock_vol()


num_moving_average = streamlit.slider("Select number of days to calculate moving average ", 2, 100)
def plot_stock_moving_average(num_days):
    fig = graph_objs.Figure()
    fig.add_trace(graph_objs.Scatter(x=data["Date"], y=data['Close'].rolling(num_days).mean(), name =f"Moving average {num_days} days",line=dict(color="#32AEFA")))
    fig.layout.update(title_text="Moving Average", xaxis_rangeslider_visible=True)
    streamlit.plotly_chart(fig)
plot_stock_moving_average(num_moving_average)




num_years = streamlit.slider("Select the number of months to be forecasted:", 1, 20)
period = num_years * 31

df_train = data[["Date","Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close":"y"})

model = prophet.Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

streamlit.subheader("Forecast Data")
streamlit.write(forecast.tail())

# streamlit.write("Forecast data visualised")
fig1 = plot_plotly(model, forecast)
fig1.layout.update(title_text="Forecast data visualised", xaxis_rangeslider_visible=True)
streamlit.plotly_chart(fig1)

# streamlit.write('Forecast components')
fig2 = model.plot_components(forecast)
fig1.layout.update(title_text="Forecast components", xaxis_rangeslider_visible=True)
streamlit.write(fig2)

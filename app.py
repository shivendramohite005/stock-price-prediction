# app.py - Complete Fixed Version (Without ExtraTrees)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Stock Prediction – Indian Stocks", layout="wide")
st.title("📈 Advanced Stock Prediction – Indian Stocks")

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_yfinance(ticker, period="2y", interval="1d"):
    """Fetch stock data from Yahoo Finance"""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False).reset_index()
        hist.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        hist = hist[['date', 'open', 'high', 'low', 'close', 'volume']]
        hist['date'] = pd.to_datetime(hist['date'])
        
        if len(hist) < 50:
            raise ValueError("Insufficient data. Try a longer period.")
        
        return hist
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def add_indicators(df):
    """Add technical indicators to dataframe"""
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # Moving Averages
    df['sma_10'] = df['close'].rolling(window=10, min_periods=10).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    
    # RSI
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi_indicator.rsi()
    
    # MACD
    macd_indicator = MACD(close=df['close'])
    df['macd'] = macd_indicator.macd()
    
    # Bollinger Bands
    bb_indicator = BollingerBands(close=df['close'])
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def create_lag_features(df, lags=[1, 2, 3]):
    """Create lagged features for time series"""
    df = df.copy()
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df.dropna(inplace=True)
    return df

def build_sequences_multivariate(X, y, seq_len):
    """Build sequences for LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

def prepare_forecast_input(last_data, scaler, features):
    """Prepare input for forecasting"""
    last_scaled = scaler.transform(last_data[features].values)
    return last_scaled

# ---------------------------
# Sidebar Configuration
# ---------------------------
st.sidebar.header("📊 Data Source Configuration")
ticker = st.sidebar.text_input("Ticker (e.g., TCS.NS, RELIANCE.NS)", value="TCS.NS")
period = st.sidebar.selectbox("Period", ("6mo", "1y", "2y", "5y"), index=2)
interval = st.sidebar.selectbox("Interval", ("1d", "1wk"), index=0)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if st.sidebar.button("🔄 Fetch Data"):
    with st.spinner("Fetching data from Yahoo Finance..."):
        try:
            stock_df = fetch_yfinance(ticker, period=period, interval=interval)
            stock_df.sort_values('date', inplace=True)
            stock_df.set_index('date', inplace=True)
            
            # Store in session state
            st.session_state.stock_df = stock_df
            st.session_state.chosen_symbol = ticker.upper()
            st.session_state.data_loaded = True
            
            st.sidebar.success(f"✅ Fetched {len(stock_df)} rows for {ticker.upper()}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")
            st.session_state.data_loaded = False
            st.stop()

if not st.session_state.data_loaded:
    st.info("👈 Please click 'Fetch Data' in the sidebar to load stock data.")
    st.stop()

# Retrieve data from session state
stock_df = st.session_state.stock_df
chosen_symbol = st.session_state.chosen_symbol

# ---------------------------
# Feature Engineering
# ---------------------------
with st.spinner("Processing indicators and features..."):
    stock = add_indicators(stock_df)
    stock = create_lag_features(stock, lags=[1, 2, 3])
    stock = stock.sort_index()

# ---------------------------
# Create Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 EDA", "🔍 Indicators", "🤖 Prediction", "📄 Report"])

# ---------------------------
# TAB 1: Overview
# ---------------------------
with tab1:
    st.subheader(f"Overview – {chosen_symbol}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = go.Figure(data=[go.Candlestick(
            x=stock.index,
            open=stock['open'],
            high=stock['high'],
            low=stock['low'],
            close=stock['close'],
            name='OHLC'
        )])
        fig.update_layout(
            template="plotly_dark",
            title=f"Candlestick Chart – {chosen_symbol}",
            height=520,
            xaxis_title="Date",
            yaxis_title="Price"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        last = stock[['open', 'high', 'low', 'close', 'volume']].iloc[-1]
        prev = stock['close'].iloc[-2]
        change = last['close'] - prev
        pct_change = (change / prev) * 100
        
        st.metric("Close Price", f"₹{last['close']:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
        st.metric("Volume", f"{int(last['volume']):,}")
        st.metric("High", f"₹{last['high']:.2f}")
        st.metric("Low", f"₹{last['low']:.2f}")
    
    st.subheader("Recent Data")
    st.dataframe(stock[['open', 'high', 'low', 'close', 'volume']].tail(10).style.format("{:.2f}"))

# ---------------------------
# TAB 2: EDA
# ---------------------------
with tab2:
    st.subheader("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Summary Statistics**")
        stats_df = stock[['open', 'high', 'low', 'close', 'volume']].describe()
        st.dataframe(stats_df.style.format("{:.2f}"))
    
    with col2:
        st.write("**Price Information**")
        st.metric("Data Points", len(stock))
        st.metric("Date Range", f"{stock.index[0].strftime('%Y-%m-%d')} to {stock.index[-1].strftime('%Y-%m-%d')}")
        st.metric("Average Close", f"₹{stock['close'].mean():.2f}")
        st.metric("Std Deviation", f"₹{stock['close'].std():.2f}")
    
    # Price Distribution
    st.subheader("Close Price Distribution")
    fig_hist = px.histogram(
        stock, 
        x='close', 
        nbins=50, 
        template="plotly_dark",
        title="Distribution of Close Prices"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Volume over time
    st.subheader("Volume Over Time")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=stock.index, y=stock['volume'], name='Volume'))
    fig_vol.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_vol, use_container_width=True)

# ---------------------------
# TAB 3: Indicators
# ---------------------------
with tab3:
    st.subheader("🔍 Technical Indicators")
    
    ind = st.selectbox("Select Indicator", ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"])
    
    fig = go.Figure()
    
    if ind == "SMA":
        fig.add_trace(go.Scatter(x=stock.index, y=stock['close'], name='Close', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=stock.index, y=stock['sma_10'], name='SMA 10', line=dict(color='orange')))
        fig.update_layout(title="Simple Moving Average (10)", template="plotly_dark")
    
    elif ind == "EMA":
        fig.add_trace(go.Scatter(x=stock.index, y=stock['close'], name='Close', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=stock.index, y=stock['ema_10'], name='EMA 10', line=dict(color='orange')))
        fig.update_layout(title="Exponential Moving Average (10)", template="plotly_dark")
    
    elif ind == "RSI":
        fig.add_trace(go.Scatter(x=stock.index, y=stock['rsi'], name='RSI', line=dict(color='purple')))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title="Relative Strength Index (RSI)", template="plotly_dark")
    
    elif ind == "MACD":
        fig.add_trace(go.Scatter(x=stock.index, y=stock['macd'], name='MACD', line=dict(color='blue')))
        fig.update_layout(title="MACD Indicator", template="plotly_dark")
    
    elif ind == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=stock.index, y=stock['close'], name='Close', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=stock.index, y=stock['bb_upper'], name='Upper Band', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=stock.index, y=stock['bb_middle'], name='Middle Band', line=dict(color='yellow', dash='dot')))
        fig.add_trace(go.Scatter(x=stock.index, y=stock['bb_lower'], name='Lower Band', line=dict(color='green', dash='dash')))
        fig.update_layout(title="Bollinger Bands", template="plotly_dark")
    
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TAB 4: Prediction
# ---------------------------
with tab4:
    st.subheader("🤖 Price Prediction & Forecasting")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox("Model", ["LinearRegression", "RandomForest", "KNN", "XGBoost", "LSTM"])
    with col2:
        forecast_days = st.number_input("Forecast Days", 1, 60, 7)
    with col3:
        test_size = st.slider("Test Size (%)", 5, 40, 20)
    
    run_pred = st.button("🚀 Run Prediction", type="primary")
    
    features = ['close', 'sma_10', 'ema_10', 'rsi', 'macd', 'bb_upper', 'close_lag_1', 'close_lag_2', 'close_lag_3']
    
    if run_pred:
        try:
            # Prepare data
            df_model = stock[features].copy()
            df_model['target'] = stock['close'].shift(-1)
            df_model.dropna(inplace=True)
            
            if len(df_model) < 100:
                st.error("Not enough data for prediction. Try a longer period.")
                st.stop()
            
            # ---------------------------
            # LSTM MODEL
            # ---------------------------
            if model_name == "LSTM":
                st.info("Training LSTM model... This may take a few minutes.")
                
                X = df_model[features].values
                y = df_model['target'].values
                
                # Scale the data
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
                
                # Create sequences
                seq_len = min(10, len(X_scaled) // 10)
                X_seq, y_seq = build_sequences_multivariate(X_scaled, y_scaled, seq_len)
                
                # Split data
                split_idx = int(len(X_seq) * (1 - test_size/100.0))
                X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_len, X_train.shape[2])),
                    Dropout(0.2),
                    LSTM(50, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                
                # Train
                with st.spinner("Training LSTM..."):
                    history = model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        verbose=0,
                        validation_split=0.1
                    )
                
                # Predict
                preds_scaled = model.predict(X_test, verbose=0)
                preds = scaler_y.inverse_transform(preds_scaled).flatten()
                y_test_actual = scaler_y.inverse_transform(y_test).flatten()
                
                # Metrics
                r2 = r2_score(y_test_actual, preds)
                mae = mean_absolute_error(y_test_actual, preds)
                rmse = math.sqrt(mean_squared_error(y_test_actual, preds))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.4f}")
                col2.metric("MAE", f"₹{mae:.2f}")
                col3.metric("RMSE", f"₹{rmse:.2f}")
                
                # Plot Actual vs Predicted
                st.subheader("Actual vs Predicted Prices")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test_actual, mode='lines', name='Actual', line=dict(color='cyan')))
                fig.add_trace(go.Scatter(y=preds, mode='lines', name='Predicted', line=dict(color='orange')))
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast
                st.subheader(f"Forecast for Next {forecast_days} Days")
                last_seq = X_scaled[-seq_len:]
                future_preds = []
                future_dates = [stock.index[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
                
                for _ in range(forecast_days):
                    pred_scaled = model.predict(last_seq.reshape(1, seq_len, -1), verbose=0)
                    pred_price = scaler_y.inverse_transform(pred_scaled)[0][0]
                    future_preds.append(pred_price)
                    
                    # Update sequence
                    new_row = last_seq[-1].copy()
                    new_row[0] = pred_scaled[0][0]
                    last_seq = np.vstack([last_seq[1:], new_row])
                
                forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})
                st.dataframe(forecast_df.style.format({"Predicted Price": "₹{:.2f}"}))
                
                # Visualize forecast
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=stock.index[-30:], 
                    y=stock['close'].tail(30), 
                    name='Historical', 
                    line=dict(color='cyan')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_preds, 
                    name='Forecast', 
                    line=dict(color='orange', dash='dash')
                ))
                fig_forecast.update_layout(template="plotly_dark", title="Price Forecast", height=400)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.download_button(
                    "📥 Download Forecast CSV",
                    forecast_df.to_csv(index=False).encode(),
                    file_name=f"{chosen_symbol}_forecast.csv",
                    mime="text/csv"
                )
            
            # ---------------------------
            # ML MODELS
            # ---------------------------
            else:
                X = df_model[features].values
                y = df_model['target'].values
                
                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split
                split_idx = int(len(X_scaled) * (1 - test_size/100.0))
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Select model
                if model_name == "LinearRegression":
                    model = LinearRegression()
                elif model_name == "RandomForest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                elif model_name == "KNN":
                    model = KNeighborsRegressor(n_neighbors=5)
                elif model_name == "XGBoost":
                    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                
                # Train
                with st.spinner(f"Training {model_name}..."):
                    model.fit(X_train, y_train)
                
                # Predict
                preds = model.predict(X_test)
                
                # Metrics
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                rmse = math.sqrt(mean_squared_error(y_test, preds))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.4f}")
                col2.metric("MAE", f"₹{mae:.2f}")
                col3.metric("RMSE", f"₹{rmse:.2f}")
                
                # Plot Actual vs Predicted
                st.subheader("Actual vs Predicted Prices")
                comparison_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': preds
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual', line=dict(color='cyan')))
                fig.add_trace(go.Scatter(y=preds, mode='lines', name='Predicted', line=dict(color='orange')))
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast
                st.subheader(f"Forecast for Next {forecast_days} Days")
                
                last_known = df_model[features].iloc[-1:].copy()
                future_preds = []
                future_dates = [stock.index[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
                
                current_features = last_known.values.copy()
                
                for i in range(forecast_days):
                    # Scale and predict
                    current_scaled = scaler.transform(current_features)
                    pred_price = model.predict(current_scaled)[0]
                    future_preds.append(pred_price)
                    
                    # Update features for next prediction
                    # Shift lags
                    current_features[0, 8] = current_features[0, 7]  # lag_3 = lag_2
                    current_features[0, 7] = current_features[0, 6]  # lag_2 = lag_1
                    current_features[0, 6] = current_features[0, 0]  # lag_1 = close
                    current_features[0, 0] = pred_price  # close = prediction
                    
                    # Simple moving average update
                    current_features[0, 1] = pred_price * 0.7 + current_features[0, 1] * 0.3
                    current_features[0, 2] = pred_price * 0.7 + current_features[0, 2] * 0.3
                
                forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})
                st.dataframe(forecast_df.style.format({"Predicted Price": "₹{:.2f}"}))
                
                # Visualize forecast
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=stock.index[-30:], 
                    y=stock['close'].tail(30), 
                    name='Historical', 
                    line=dict(color='cyan')
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_preds, 
                    name='Forecast', 
                    line=dict(color='orange', dash='dash')
                ))
                fig_forecast.update_layout(template="plotly_dark", title="Price Forecast", height=400)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.download_button(
                    "📥 Download Forecast CSV",
                    forecast_df.to_csv(index=False).encode(),
                    file_name=f"{chosen_symbol}_forecast.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.exception(e)

# ---------------------------
# TAB 5: Report
# ---------------------------
with tab5:
    st.subheader("📄 Generate PDF Report")
    
    st.info("Generate a comprehensive PDF report with stock analysis and statistics.")
    
    if st.button("📄 Generate PDF Report"):
        try:
            with st.spinner("Generating PDF..."):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, f"{chosen_symbol} Stock Analysis Report", ln=True, align='C')
                
                pdf.set_font("Arial", size=10)
                pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                pdf.ln(10)
                
                # Summary Statistics
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Summary Statistics", ln=True)
                pdf.set_font("Courier", size=8)
                
                stats = stock[['open', 'high', 'low', 'close', 'volume']].describe()
                stats_text = stats.to_string()
                
                for line in stats_text.split('\n'):
                    pdf.cell(0, 5, line, ln=True)
                
                pdf.ln(5)
                
                # Recent Prices
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 8, "Recent Prices (Last 5 Days)", ln=True)
                pdf.set_font("Courier", size=8)
                
                recent = stock[['open', 'high', 'low', 'close']].tail(5)
                recent_text = recent.to_string()
                
                for line in recent_text.split('\n'):
                    pdf.cell(0, 5, line, ln=True)
                
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                
                st.download_button(
                    "📥 Download PDF Report",
                    pdf_bytes,
                    file_name=f"{chosen_symbol}_report.pdf",
                    mime="application/pdf"
                )
                
                st.success("✅ PDF report generated successfully!")
                
        except Exception as e:
            st.error(f"❌ Error generating PDF: {str(e)}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Advanced Stock Prediction App | Data from Yahoo Finance</p>
        <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)
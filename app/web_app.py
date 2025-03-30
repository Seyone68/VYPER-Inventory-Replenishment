import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping
import time

import warnings
warnings.filterwarnings('ignore')

# Set the title of the web app
st.set_page_config(page_title='VYPER Stock Replenishment', page_icon=':bar_chart:', layout='wide')

# Set the favicon
favicon = """
<link rel="shortcut icon" href="data:image/x-icon;," type="image/x-icon">
"""
st.markdown(favicon, unsafe_allow_html=True)

# Set the background color of the web app
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f5f5
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the color of the sidebar menu
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        color: #f0f5f5
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the color of the sidebar menu items
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #333333 !important;  /* Dark gray */
        color: white !important;  /* White text */
    }
    
    /* Sidebar text color */
    .sidebar .sidebar-content {
        color: white !important;
    }

    /* Title text color */
    h1, h2, h3, h4, h5, h6 {
        color: #da031f;  /* Red */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the color of the main content
st.markdown(
    """
    <style>
    /* Background color of the main content */
    .stApp {
        background-color: #d3d3d3;  /* Light gray */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the color of the main content title
st.markdown(
    """
    <style>
    .main h1, .main h2 {
        color: #4b168e
    }
    .main h3, .main h4, .main h5, .main h6 {
        color: #073496
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the color of the main content text
st.markdown(
    """
    <style>
    .main p {
        color: #079669
    }
    .main a {
        color: #ffffff
    }
    /* Change button styling */
    .stDownloadButton button {
        background-color: #f18d8e !important;  /* Green button */
        font-color: #ffffff !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit App title
st.title('VYPER Stock Replenishment 📊')
st.markdown('An advanced **machine learning** to predict stock replenishment')

# Sidebar for uploading data
st.sidebar.title('📂 Upload Data')
sales_file = st.sidebar.file_uploader("📌 Upload Sales Data (CSV or Excel)", type=["csv", "xlsx"])
stock_file = st.sidebar.file_uploader("📌 Upload Stock Data (CSV or Excel)", type=["csv", "xlsx"])

# Button to Load Data
load_data = st.sidebar.button("🔄 Load Data")

if load_data:
    if sales_file and stock_file:
        progress_bar = st.sidebar.progress(0)  # Initialize Progress Bar
        
        # Load datasets
        st.sidebar.write("📥 Loading files...")
        time.sleep(1)  # Simulate loading delay
        
        df_sales = pd.read_csv(sales_file) if sales_file.name.endswith("csv") else pd.read_excel(sales_file)
        df_stock = pd.read_csv(stock_file, skiprows=3) if stock_file.name.endswith("csv") else pd.read_excel(stock_file, skiprows=3)
        
        progress_bar.progress(20)  # Update Progress
        
        # Preprocess Sales Data
        df_sales.dropna(axis=1, how='all', inplace=True)
        # Remove '12:35:57+00:00' from 'Date' column
        df_sales['Date'] = pd.to_datetime(df_sales['Date'], format='mixed', errors='coerce')
        df_sales['Date'] = df_sales['Date'].astype(str)
        df_sales['Date'] = df_sales['Date'].str.split(' ').str[0]
        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        df_sales.drop(columns=['Vyper ID', 'Shipping Label', 'Dispatch Date', 'Currency', 'Delivery Amount', 
                        'Discount_amount','Payment Status', 'Invoice Status', 'Marketplace', 'Online Order Id','Product Brand'], inplace=True,
                        errors='ignore')
        
        # Preprocess Stock Data
        df_stock.drop(df_stock.tail(1).index, inplace=True)
        df_stock.dropna(axis=1, how='all', inplace=True)
        df_stock = df_stock.astype({'Initial Stock': 'int64','Stock In': 'int64', 'Stock Out': 'int64', 'Stock On Hand': 'int64'})
        df_stock.drop(columns=['Brand Name', 'Initial Stock', 'Stock In', 'Stock Out'], inplace=True, errors='ignore')
        # Validate and clean the column names
        df_stock.rename(columns=lambda x: x.strip(), inplace=True)

        st.write("### Sales Data Preview:")
        st.dataframe(df_sales.head(15))
        
        st.write("### Stock Data Preview:")
        st.dataframe(df_stock.tail(15))

        # Aggregate Sales Data to Daily level
        daily_sales = df_sales.groupby('Date')['QTY'].sum().reset_index()
        daily_sales['Rolling_Mean_7'] = daily_sales['QTY'].rolling(window=7).mean()
        daily_sales['Rolling_Std_7'] = daily_sales['QTY'].rolling(window=7).std()
        daily_sales['Lag_1'] = daily_sales['QTY'].shift(1)
        daily_sales['Lag_7'] = daily_sales['QTY'].shift(7)
        daily_sales.fillna(0, inplace=True)
        daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
        daily_sales['Month'] = daily_sales['Date'].dt.month

        # Visualize Sold Quantity by type
        st.subheader('Sold Quantity by Type')
        plt.figure(figsize=(15, 8))
        sns.barplot(x='Type', y='QTY', data=df_sales, ci=None, estimator=sum, color='#2980b9')
        # Display value on top of each bar
        for p in plt.gca().patches:
            plt.gca().annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                            textcoords='offset points')
        plt.title('Sold Quantity by Type')
        plt.xlabel('Marketplace')
        plt.ylabel('Quantity')
        st.pyplot(plt)

        # Visualize Sales by Month - 'month-year' format in ascending order
        df_sales['Month_Year'] = df_sales['Date'].dt.to_period('M')
        st.subheader('Sales by Month')
        plt.figure(figsize=(15, 8))
        sns.barplot(x=df_sales['Month_Year'], y='Full Total', data=df_sales, ci=None, estimator=sum, color='#9575cd', order=sorted(df_sales['Month_Year']))
        # Display value on top of each bar
        for p in plt.gca().patches:
            plt.gca().annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                            textcoords='offset points')
        plt.title('Sales by Month')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        st.pyplot(plt)

        # Visualize Top 5 products by Sales
        st.subheader('Top 5 Products by Sales')
        fig, ax = plt.subplots(figsize=(15, 8))
        top_5_sales = sns.barplot(data=df_sales.groupby('Product Name')['Full Total'].sum().sort_values(ascending=False).head(5).reset_index(), y='Product Name', x='Full Total', palette='magma')
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}", (p.get_width(), p.get_y() + p.get_height() / 2.), ha='center', va='center', fontsize=12, color='black', xytext=(20, 0), textcoords='offset points')
            
        plt.title('Top 5 Products by Sales', fontsize=18)
        plt.xlabel('Total Sales', fontsize=14)
        plt.ylabel('Product Name', fontsize=14)
        st.pyplot(plt)

        # Visualize Top 5 products by Quantity
        st.subheader('Top 5 Products by Quantity')
        fig, ax = plt.subplots(figsize=(15, 8))
        top_5_qty = sns.barplot(data=df_sales.groupby('Product Name')['QTY'].sum().sort_values(ascending=False).head(5).reset_index(), y='Product Name', x='QTY', palette='rocket')
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width()):,}", (p.get_width(), p.get_y() + p.get_height() / 2.), ha='center', va='center', fontsize=12, color='black', xytext=(20, 0), textcoords='offset points')
        plt.title('Top 5 Products by Quantity', fontsize=18)
        plt.xlabel('Total Quantity', fontsize=14)
        plt.ylabel('Product Name', fontsize=14)
        st.pyplot(plt)

        # Normalize the data using StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(daily_sales[['QTY', 'Rolling_Mean_7', 'Rolling_Std_7', 'Lag_1', 'Lag_7', 'DayOfWeek', 'Month']])
        
        # Prepare sequences for BiLSTM and Gradient Boosting models
        sequence_length = 14
        num_features = scaled_data.shape[1]

        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length, 0])

        X, y = np.array(X), np.array(y)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=540, input_shape=(sequence_length, num_features), return_sequences=False))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(64, activation='relu'))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(1))  # Output layer
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Make predictions
        lstm_test_pred = lstm_model.predict(X_test)

        # Define CNN model
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=180, kernel_size=3, activation='relu', input_shape=(sequence_length, num_features)))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(64, activation='relu'))
        cnn_model.add(Dense(1))
        cnn_model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the cnn_model
        cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Make predictions
        cnn_test_pred = cnn_model.predict(X_test)
        
        # Stack LSTM and CNN predictions
        stacked_features = np.column_stack((lstm_test_pred.flatten(), cnn_test_pred.flatten()))
        
        # Train Final Regressor (RandomForest)
        final_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        final_regressor.fit(stacked_features, y_test)
        final_predictions = final_regressor.predict(stacked_features)

        progress_bar.progress(100)  # Complete Progress
        st.sidebar.success("✅ Prediction Complete!")

        # Forecast next 30 days
        future_predictions = []
        last_sequence = scaled_data[-sequence_length:]

        for _ in range(30):
            # Predict the next value using LSTM and CNN models
            lstm_next_pred = lstm_model.predict(last_sequence.reshape(1, sequence_length, num_features))
            cnn_next_pred = cnn_model.predict(last_sequence.reshape(1, sequence_length, num_features))

            # Combine the predictions using column stacking
            stacked_next_pred = np.column_stack((lstm_next_pred.flatten(), cnn_next_pred.flatten()))
            final_next_pred = final_regressor.predict(stacked_next_pred)

            # Store the final prediction
            future_predictions.append(final_next_pred)

            # Update the sequence
            next_input = np.concatenate([last_sequence[-1, :-1], final_next_pred])
            last_sequence = np.vstack([last_sequence[1:], next_input])

        # Convert the future predictions to a single array
        future_predictions = np.array(future_predictions).reshape(-1, 1)

        # Concatenate zeros to match the original feature shape for inverse transformation
        future_predictions_full = np.column_stack([future_predictions, np.zeros((future_predictions.shape[0], num_features - 1))])

        # Inverse transform the predictions
        future_predictions_rescaled = scaler.inverse_transform(future_predictions_full)
        future_predictions_units = future_predictions_rescaled[:, 0]

        # Visualization
        st.write("### Replenishment Trend:")
        plt.figure(figsize=(15, 8))
        plt.plot(daily_sales['Date'], daily_sales['QTY'], label='Actual Sales', color='#168e72')
        future_dates = pd.date_range(start=daily_sales['Date'].max() + pd.Timedelta(days=1), periods=30, freq='D')
        plt.plot(future_dates, future_predictions_units, label='Forecasted Sales', linestyle='--', color='red')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Sales Quantity', fontsize=14)
        plt.title('Sales Forecast', fontsize=18)
        plt.legend()
        st.pyplot(plt)
        
        # Restock Recommendations
        st.subheader('Restock Recommendations')
        last_30_days_sales = df_sales[df_sales['Date'] >= df_sales['Date'].max() - pd.Timedelta(days=30)]
        last_30_days_sales = last_30_days_sales.groupby(['SKU','Product Name'])['QTY'].sum().reset_index()
        last_30_days_sales.columns = ['SKU', 'Product Name', 'Last 30 Days Sales']
        current_stock_levels = df_stock.groupby('SKU')['Stock On Hand'].sum().reset_index()
        current_stock_levels.columns = ['SKU', 'Current Stock']
        restock_skus = pd.merge(last_30_days_sales, current_stock_levels, on='SKU', how='inner')
        average_daily_forecast = future_predictions_units.mean()
        restock_skus['Restock Quantity'] = restock_skus.apply(lambda x: max(0, x['Last 30 Days Sales'] - x['Current Stock'] + average_daily_forecast), axis=1)
        restock_skus['Restock Quantity'] = restock_skus['Restock Quantity'].apply(np.ceil).astype(int)
        restock_skus['Restock Quantity'] = restock_skus['Restock Quantity'].apply(lambda x: x + (6 - x % 6) if x % 6 != 0 else x)
        restock_skus['Restock Quantity'] = restock_skus.apply(lambda x: 0 if x['Current Stock'] >= x['Last 30 Days Sales'] else x['Restock Quantity'], axis=1)
        # Restock quantity not required for SKUs start with 'UKRFB
        restock_skus = restock_skus[~restock_skus['SKU'].str.startswith('UKRFB')]
        # Restock quantity not required for SKUs start with 'DERFB'
        restock_skus = restock_skus[~restock_skus['SKU'].str.startswith('DERFB')]
        # Restock quantity not required for SKUs ends with '- R'
        restock_skus = restock_skus[~restock_skus['SKU'].str.endswith('- R')].reset_index(drop=True)
        
        # Highlight the restock quantity
        def highlight_restock_qty(x):
            color = 'background-color: #2980b9' if x['Restock Quantity'] > 0 else ''
            return [color for _ in x]
        
        # Display the restock recommendations
        st.dataframe(restock_skus[['SKU', 'Product Name', 'Last 30 Days Sales','Current Stock', 'Restock Quantity']].style.apply(highlight_restock_qty, axis=1))

        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        restock_skus.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()  # Convert StringIO to a string

        # Fix: Convert string to bytes before passing to download_button
        st.download_button(label="Download Restock Recommendations", 
                        data=csv_string.encode('utf-8'),  # Convert string to bytes
                        file_name="restock_recommendations.csv", 
                        mime="text/csv")
        
        # Final Message
        st.success("🎯 The model has successfully predicted the replenishment trends!")
    else:
        st.warning("⚠️ Please upload both Sales and Stock data files to proceed!")
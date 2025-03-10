import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Streamlit App title
st.title('VYPER Stock Replenishment')
st.markdown('An advanced machine learning model to predict stock replenishment')

# Sidebar for uploading data
st.sidebar.title('Upload Data')
sales_file = st.sidebar.file_uploader("Upload Sales Data (Excel)", type=['xlsx'])
stock_file = st.sidebar.file_uploader("Upload Stock Data (Excel)", type=['xlsx'])

if sales_file and stock_file:
    # Load sales data & stock data
    df_sales = pd.read_excel(sales_file)
    df_stock = pd.read_excel(stock_file)

    # Preprocess Sales Data
    df_sales.dropna(axis=1, how='all', inplace=True)
    df_sales['Date'] = pd.to_datetime(df_sales['Date'], format='%Y-%m-%d', errors='coerce')
    #df_sales['Date'] = pd.to_datetime(df_sales['Date'].str.split(' ').str[0])
    df_sales.drop(columns=['Vyper ID', 'Shipping Label', 'Dispatch Date', 'Currency', 'Delivery Amount', 
                       'Discount_amount','Payment Status', 'Invoice Status', 'Marketplace', 'Online Order Id','Product Brand'], inplace=True,
                       errors='ignore')
    
    # Preprocess Stock Data
    df_stock.dropna(axis=1, how='all', inplace=True)
    df_stock.drop(columns=['Brand Name', 'Initial Stock', 'Stock In', 'Stock Out'], inplace=True, errors='ignore')
    # Validate and clean the column names
    df_stock.rename(columns=lambda x: x.strip(), inplace=True)

    # Ensure 'Stock On Hand' exists and convert it to int64
    if 'Stock On Hand' in df_stock.columns:
        df_stock = df_stock.astype({'Stock On Hand': 'int64'})
    else:
        st.error("The required column 'Stock On Hand' is missing from the stock data. Please upload a valid file.")

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
    sns.barplot(x='Type', y='QTY', data=df_sales, ci=None, estimator=sum, palette='coolwarm')
    # Display value on top of each bar
    for p in plt.gca().patches:
        plt.gca().annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                        textcoords='offset points')
    plt.title('Sold Quantity by Type', fontsize=20)
    plt.xlabel('Marketplace', fontsize=15)
    plt.ylabel('Quantity', fontsize=15)
    st.pyplot(plt)

    # Visualize Sales by Month
    st.subheader('Sales by Month')
    plt.figure(figsize=(15, 8))
    sns.barplot(x=df_sales['Date'].dt.month, y='Full Total', data=df_sales, ci=None, estimator=sum, palette='coolwarm')
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
    top_5_qty = sns.barplot(data=df_sales.groupby('Product Name')['QTY'].sum().sort_values(ascending=False).head(5).reset_index(), y='Product Name', x='QTY', palette='rocket')
    for p in ax.patches:
        ax.annotate(f"{int(p.get_width()):,}", (p.get_width(), p.get_y() + p.get_height() / 2.), ha='center', va='center', fontsize=12, color='black', xytext=(20, 0), textcoords='offset points')
    plt.title('Top 5 Products by Quantity', fontsize=18)
    plt.xlabel('Total Quantity', fontsize=14)
    plt.ylabel('Product Name', fontsize=14)
    st.pyplot(plt)

    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(daily_sales[['QTY', 'Rolling_Mean_7', 'Rolling_Std_7', 'Lag_1', 'Lag_7', 'DayOfWeek', 'Month']])

    # Prepare Sequence for Model
    sequence_length = 14
    num_features = scaled_data.shape[1]
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, 0])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Build & Train LSTM Model
    def build_lstm_model(hp):
        model = Sequential()
        # Add LSTM layer
        model.add(LSTM(units=hp.Int('units', min_value=64, max_value=512, step=32),
                    input_shape=(sequence_length, num_features), return_sequences=False))
        model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        # Add Dense layer
        model.add(Dense(units=hp.Int('dense_units', min_value=64, max_value=512, step=32), activation='relu'))
        model.add(Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1)))
        # Output layer
        model.add(Dense(1))
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    tuner_lstm = kt.RandomSearch(build_lstm_model, objective='val_loss', max_trials=12, executions_per_trial=4, directory='lstm_tuner', project_name='lstm_model', overwrite=True)
    tuner_lstm.search(X_train, y_train, epochs=50, validation_split=0.2)
    # Retrieve the best hyperparameters
    best_hps_lstm = tuner_lstm.get_best_hyperparameters()[0]
    # Build the best model using the best hyperparameters
    best_lstm_model = tuner_lstm.hypermodel.build(best_hps_lstm)
    best_lstm_model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)], batch_size=32)

    # Build & Train CNN Model
    def build_cnn_model(hp):
        model = Sequential()
        # Add Conv1D layer
        model.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=512, step=32),
                        kernel_size=hp.Int('kernel_size', min_value=2, max_value=5, step=32),
                        activation='relu', input_shape=(sequence_length, num_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=hp.Int('filters2', min_value=32, max_value=512, step=32),
                        kernel_size=hp.Int('kernel_size2', min_value=2, max_value=5, step=32),
                        activation='relu', input_shape=(sequence_length, num_features)))
        model.add(MaxPooling1D(pool_size=2))
        # Flatten the output
        model.add(Flatten())
        # Add Dense layer
        model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=512, step=32), activation='relu'))
        model.add(Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1)))
        # Output layer
        model.add(Dense(1))
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    tuner_cnn = kt.RandomSearch(build_cnn_model, objective='val_loss', max_trials=12, executions_per_trial=2, directory='cnn_tuner', project_name='cnn_model', overwrite=True)
    tuner_cnn.search(X_train, y_train, epochs=50, validation_split=0.2)
    best_hps_cnn = tuner_cnn.get_best_hyperparameters()[0]
    # Build the best model using the best hyperparameters
    best_cnn_model = tuner_cnn.hypermodel.build(best_hps_cnn)
    best_cnn_model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)], batch_size=32)

    # Make Predictions
    lstm_predictions = best_lstm_model.predict(X_test)
    cnn_predictions = best_cnn_model.predict(X_test)
    stacked_predictions = np.column_stack((lstm_predictions.flatten(), cnn_predictions.flatten()))
    final_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    final_regressor.fit(stacked_predictions, y_test)
    final_predictions = final_regressor.predict(stacked_predictions)

    # Forecast next 30 days
    future_predictions = []
    last_sequence = scaled_data[-sequence_length:]
    for i in range(30):
        lstm_next_pred = best_lstm_model.predict(last_sequence.reshape(1, sequence_length, num_features))
        cnn_next_pred = best_cnn_model.predict(last_sequence.reshape(1, sequence_length, num_features))
        stacked_next_pred = np.column_stack((lstm_next_pred.flatten(), cnn_next_pred.flatten()))
        final_next_pred = final_regressor.predict(stacked_next_pred)
        future_predictions.append(final_next_pred)
        next_input = np.concatenate([last_sequence[-1, :-1], final_next_pred])
        last_sequence = np.vstack([last_sequence[1:], next_input])

    # Convert Predictions
    future_predictions = np.array(future_predictions).flatten()
    future_predictions_scaled = scaler.inverse_transform(np.column_stack([future_predictions, np.zeros((len(future_predictions), num_features - 1))]))[:, 0]

    # Visualize Forecast
    st.subheader('Forecasted Sales for Next 30 Days')
    plt.figure(figsize=(10, 5))
    plt.plot(daily_sales['Date'], daily_sales['QTY'], label='Actual Sales', color='#168e72')
    future_dates = pd.date_range(start=daily_sales['Date'].max() + pd.Timedelta(days=1), periods=30, freq='D')
    plt.plot(future_dates, future_predictions_scaled, label='Forecasted Sales', linestyle='--', color='red')
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.title('Sales Forecast')
    plt.legend()
    st.pyplot(plt)

    # Evaluate Model
    st.subheader('Model Evaluation')
    mae = mean_absolute_error(y_test, final_predictions)
    mse = mean_squared_error(y_test, final_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, final_predictions)
    st.write('Mean Absolute Error:', mae)
    st.write('Mean Squared Error:', mse)
    st.write('Root Mean Squared Error:', rmse)
    st.write('R2 Score:', r2)

    # Restock Recommendations
    st.subheader('Restock Recommendations')
    last_30_days_sales = df_sales[df_sales['Date'] >= df_sales['Date'].max() - pd.Timedelta(days=30)]
    last_30_days_sales = last_30_days_sales.groupby('SKU','Product Name')['QTY'].sum().reset_index()
    last_30_days_sales.columns = ['SKU', 'Product Name', 'Last 30 Days Sales']
    current_stock_levels = df_stock.groupby('SKU')['Stock On Hand'].sum().reset_index()
    current_stock_levels.columns = ['SKU', 'Current Stock']
    restock_skus = pd.merge(last_30_days_sales, current_stock_levels, on='SKU', how='inner')
    average_daily_forecast = future_predictions_scaled.mean()
    restock_skus['Restock Quantity'] = restock_skus.apply(lambda x: max(0, x['Last 30 Days Sales'] - x['Current Stock'] + average_daily_forecast), axis=1)
    restock_skus['Restock Quantity'] = restock_skus['Restock Quantity'].astype(np.ceil).astype(int)
    restock_skus['Restock Quantity'] = restock_skus['Restock_Quantity'].apply(lambda x: x + (6 - x % 6) if x % 6 != 0 else x)
    restock_skus['Restock_Quantity'] = restock_skus.apply(lambda x: 0 if x['Current_Stock'] >= x['Last_30_Days_Sales'] else x['Restock_Quantity'], axis=1)
    # Restock quantity not required for SKUs start with 'UKRFB
    restock_skus = restock_skus[~restock_skus['SKU'].str.startswith('UKRFB')]
    # Restock quantity not required for SKUs start with 'DERFB'
    restock_skus = restock_skus[~restock_skus['SKU'].str.startswith('DERFB')]
    # Restock quantity not required for SKUs ends with '- R'
    restock_skus = restock_skus[~restock_skus['SKU'].str.endswith('- R')].reset_index(drop=True)
    st.dataframe(restock_skus[['SKU', 'Product Name', 'Last 30 Days Sales','Current Stock', 'Restock Quantity']])

    # Download Restock Quantities
    csv_buffer = io.StringIO()
    restock_skus.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    st.download_button(label="Download Restock Recommendations", data=csv_buffer, file_name="restock_recommendations.csv", mime="text/csv")

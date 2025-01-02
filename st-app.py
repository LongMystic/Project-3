import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

from keras.api.models import load_model

st.set_page_config("Warehouse Forecasting", layout="wide")

REQUIRED_COLUMNS = ["Unit quantity", "Weight", "Truck Count", "Daily Capacity "]


@st.cache_resource
def load_lstm_model():
    return load_model("model_lstm.h5")


@st.cache_resource
def load_rnn_model():
    return load_model("model_rnn.h5")


@st.cache_resource
def load_gru_model():
    return load_model("model_gru.h5")


model_lstm = load_lstm_model()
model_rnn = load_rnn_model()
model_gru = load_gru_model()

columns = ['Unit quantity', 'Weight', 'Truck Count', 'Order Date']
drop_columns = ['Order ID', 'Origin Port', 'Plant ID', 'Daily Capacity ', 'Plant Code', 'Destination Port', 'Carrier',
                'Customer', 'Service Level']  # 9
numeric_columns = ['TPT', 'Ship ahead day count', 'Ship Late Day count', 'Product ID', 'Unit quantity', 'Weight',
                   'Truck Count']  # 7


def visualize_with_ex(df, column_name, df_pred=None):
    df = df.tail(30)

    # Create the base figure
    fig = go.Figure()

    if column_name == 'Unit quantity':
        fig.add_trace(go.Bar(x=df['Order Date'], y=df[column_name], name=column_name))

        # Add prediction lines if provided
        if df_pred is not None:
            fig.add_trace(go.Bar(x=df_pred['Order Date'], y=df_pred[column_name],
                                 name=f'{column_name} prediction'))
    elif column_name == 'Weight':
        if df_pred is None:
            fig.add_trace(go.Scatter(x=df['Order Date'], y=df[column_name], fill='tozeroy', mode='none', name=column_name))
            fig.add_trace(go.Scatter(x=df['Order Date'], y=df['Daily Capacity '], fill='tonexty', mode='none',
                                     name='Daily Capacity'))

        # Add prediction lines if provided
        if df_pred is not None:
            fig.add_trace(
                go.Scatter(x=df_pred['Order Date'], y=df_pred['Daily Capacity'], fill='tonexty',
                           mode='none', name='Daily Capacity'))
            fig.add_trace(go.Scatter(x=df_pred['Order Date'], y=df_pred[column_name], fill='tozeroy', mode='none',
                                     name=f'{column_name} prediction'))
    else:
        fig.add_trace(go.Scatter(
            x=df['Order Date'],
            y=df[column_name],
            mode='lines+markers',
            name=column_name,
            marker=dict(symbol='circle')
        ))

        # Add prediction lines if provided
        if df_pred is not None:
            fig.add_trace(go.Scatter(
                x=df_pred['Order Date'],
                y=df_pred[column_name],
                mode='lines+markers',
                name=f'{column_name} prediction',
                marker=dict(symbol='circle')
            ))

    # Update layout for better appearance
    fig.update_layout(
        title=f'{column_name} Over Time',
        xaxis_title='Date',
        yaxis_title='Value',
        legend_title='Legend',
        xaxis=dict(tickangle=45),
        template='plotly_white'
    )

    return fig


def evaluate(avg_30, avg_5, val):
    if avg_30 < avg_5:
        if avg_30 <= val <= avg_5:
            return 'Sufficiency'
        elif val > avg_5:
            return 'Plenty'
        elif val < avg_30:
            return 'Shortage'
    else:
        if avg_5 <= val <= avg_30:
            return 'Sufficiency'
        elif val > avg_30:
            return 'Plenty'
        elif val < avg_5:
            return 'Shortage'


def predict(_df, model):
    df = _df.copy()
    # Predict the next 14 days
    print(df.columns)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df.set_index('Order Date', inplace=True)
    df = df[numeric_columns]

    average_last_30 = df[['Unit quantity', 'Weight', 'Truck Count']].iloc[-30:].mean()

    average_last_5 = df[['Unit quantity', 'Weight', 'Truck Count']].iloc[-5:].mean()

    unit_quantity_last_30 = average_last_30['Unit quantity']
    unit_quantity_last_5 = average_last_5['Unit quantity']
    weight_last_30 = average_last_5['Weight']
    weight_last_5 = average_last_5['Weight']
    truck_count_last_30 = average_last_5['Truck Count']
    truck_count_last_5 = average_last_5['Truck Count']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    last_sequence = scaled_data[-5:]
    predicted = []

    predict_days = 7
    for _ in range(predict_days):
        prediction = model.predict(last_sequence[np.newaxis, :, :])
        predicted.append(prediction[0])
        last_sequence = np.vstack((last_sequence[1:], prediction))

    predicted = scaler.inverse_transform(predicted)

    predicted_df = pd.DataFrame(predicted, columns=df.columns)
    predicted_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=predict_days)
    predicted_df['Unit quantity'] = predicted_df['Unit quantity'].astype(int)
    predicted_df['Weight'] = predicted_df['Weight'].astype(float)
    predicted_df['Truck Count'] = predicted_df['Truck Count'].astype(int)
    predicted_df = predicted_df.reset_index()
    predicted_df.rename(columns={'index': 'Order Date'}, inplace=True)
    unit_quantity_evaluation = []
    weight_evaluation = []
    truck_count_evaluation = []
    for i in range(len(predicted_df)):
        unit_quantity_evaluation.append(evaluate(unit_quantity_last_30, unit_quantity_last_5,
                                                 predicted_df.iloc[i]['Unit quantity']))
        weight_evaluation.append(evaluate(weight_last_30, weight_last_5,
                                          predicted_df.iloc[i]['Weight']))
        truck_count_evaluation.append(evaluate(truck_count_last_30, truck_count_last_5,
                                               predicted_df.iloc[i]['Truck Count']))
    predicted_df['Unit quantity evaluation'] = unit_quantity_evaluation
    predicted_df['Weight evaluation'] = weight_evaluation
    predicted_df['Truck Count evaluation'] = truck_count_evaluation
    predicted_df['Daily Capacity'] = _df['Daily Capacity ']
    return predicted_df


def upload_file():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

            if missing_columns:
                st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
            else:
                st.success("File uploaded successfully and contains all required columns!")
                st.session_state.df = df
                st.rerun()
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload a CSV file.")


def toggle_figure():
    st.session_state.visualize = not st.session_state.visualize


def toggle_predict():
    st.session_state.predict = not st.session_state.predict


def page_1():
    # Display the DataFrame
    st.write("### Data:")
    st.dataframe(st.session_state.df)

    # 1. ADD ROW

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload a csv file"):
            st.session_state.upload_file = True
        if st.session_state.get("upload_file", False):
            upload_file()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Visualize"):
            toggle_figure()

        if st.session_state.visualize:
            fig_uq = visualize_with_ex(st.session_state.df, "Unit quantity")
            fig_w = visualize_with_ex(st.session_state.df, "Weight")
            fig_tc = visualize_with_ex(st.session_state.df, "Truck Count")

            st.session_state.fig_uq = fig_uq
            st.session_state.fig_w = fig_w
            st.session_state.fig_tc = fig_tc

            st.write('### Visualize:')
            st.plotly_chart(st.session_state.fig_uq)
            st.plotly_chart(st.session_state.fig_w)
            st.plotly_chart(st.session_state.fig_tc)

            # st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox('Choose a model:', ['RNN', 'LSTM', 'GRU'])

    with col2:
        if st.button("Predict"):
            toggle_predict()

    if st.session_state.predict:
        if model_name == "RNN":
            model_predict = model_rnn
        elif model_name == "LSTM":
            model_predict = model_lstm
        else:
            model_predict = model_gru

        df_pred = predict(st.session_state.df, model_predict)

        pred_fig_unit_quantity = visualize_with_ex(st.session_state.df, "Unit quantity", df_pred)
        pred_fig_weight = visualize_with_ex(st.session_state.df, "Weight", df_pred)
        pred_fig_truck_count = visualize_with_ex(st.session_state.df, "Truck Count", df_pred)

        st.session_state.pred_fig_unit_quantity = pred_fig_unit_quantity
        st.session_state.pred_fig_weight = pred_fig_weight
        st.session_state.pred_fig_truck_count = pred_fig_truck_count

        st.write(f"### Prediction using {model_name} model")

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Prediction about unit quantity")
            st.plotly_chart(st.session_state.pred_fig_unit_quantity)

        with col2:
            st.dataframe(df_pred[['Order Date', 'Unit quantity', 'Unit quantity evaluation']])

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Prediction about weight")
            st.plotly_chart(st.session_state.pred_fig_weight)

        with col2:
            st.dataframe(df_pred[['Order Date', 'Weight', 'Weight evaluation']])

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Prediction about truck count")
            st.plotly_chart(st.session_state.pred_fig_truck_count)

        with col2:
            st.dataframe(df_pred[['Order Date', 'Truck Count', 'Truck Count evaluation']])


def main():
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv('./data/cleaned_data.csv')

    if 'page_2_df' not in st.session_state:
        st.session_state.page_2_df = pd.DataFrame(columns=columns)

    if 'figure' not in st.session_state:
        st.session_state.figure = ''

    if 'visualize' not in st.session_state:
        st.session_state.visualize = False

    if 'predict' not in st.session_state:
        st.session_state.predict = False

    if 'upload_file' not in st.session_state:
        st.session_state.upload_file = False

    page_1()


if __name__ == "__main__":
    main()

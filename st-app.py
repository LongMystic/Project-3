import time

import streamlit as st
import pandas as pd
import numpy as np
import pymysql.cursors
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
from validator import validate_date, validate_price, validate_warehouse_capacity, validate_truck_capacity

from keras.models import load_model

connection = pymysql.connect(
    host='localhost',
    user='root',
    password='Liquid@123',
    database='prj3',
    cursorclass=pymysql.cursors.DictCursor
)

st.set_page_config("Warehouse Forecasting", layout="wide")


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
sc = joblib.load("scaler.pkl")

columns = ['price', 'warehouse_capacity', 'truck_capacity', 'date']


def insert_row_to_db():
    sql = f"""
        INSERT INTO prj3.data (date, price, warehouse_capacity, truck_capacity)
        VALUE (
            {st.session_state.date}
            , {st.session_state.price}
            , {st.session_state.warehouse_capacity}
            , {st.session_state.truck_capacity}
        );
    """
    cursor = connection.cursor()
    cursor.execute(sql)
    connection.commit()

    return 1


def fetch_data():
    sql = """
        SELECT * FROM prj3.data;
    """
    cursor = connection.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()

    columns_df = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=columns_df)

    return df


def add_row():
    st.write("### Add New Row")  # Heading for modal
    st.session_state.date = st.text_input("Enter Date")
    st.session_state.price = st.text_input("Enter Price")
    st.session_state.warehouse_capacity = st.text_input("Enter Warehouse Capacity")
    st.session_state.truck_capacity = st.text_input("Enter Truck Capacity")

    # Save and Cancel buttons within modal
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save"):
            # validate data
            code, message = validate_date(st.session_state.date)
            if code == -1:
                st.error(message)
                return

            code, message = validate_price(st.session_state.price)
            if code == -1:
                st.error(message)
                return

            code, message = validate_warehouse_capacity(st.session_state.warehouse_capacity)
            if code == -1:
                st.error(message)
                return

            code, message = validate_truck_capacity(st.session_state.truck_capacity)
            if code == -1:
                st.error(message)
                return

            if code == 0:
                # insert_row_to_db()
                st.info("Add row successfully!")
                time.sleep(2)
                st.session_state.add_row = False
                st.rerun()  # Refresh the app to display the updated DataFrame
    with col2:
        if st.button("Cancel"):
            st.session_state.add_row = False  # Close modal without saving
            st.rerun()


def visualize(df, df_pred=None):
    df = df.tail(30)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['date'], df['warehouse_capacity'], label='warehouse_capacity', marker='o')
    ax.plot(df['date'], df['truck_capacity'], label='truck_capacity', marker='x')

    if df_pred is not None:
        ax.plot(df_pred['date'], df_pred['warehouse_capacity'], label='warehouse_capacity_prediction', marker='o')
        ax.plot(df_pred['date'], df_pred['truck_capacity'], label='truck_capacity_prediction', marker='x')

    # Adding labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Capacity Over Time')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

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


def predict(df, model, scaler1: MinMaxScaler, cur_date):
    # Predict the next 14 days
    df = df[['date', 'warehouse_capacity', 'truck_capacity']]

    average_last_30 = df[['warehouse_capacity', 'truck_capacity']].iloc[-30:].mean()

    average_last_5 = df[['warehouse_capacity', 'truck_capacity']].iloc[-5:].mean()

    average_warehouse_last_30 = average_last_30['warehouse_capacity']
    average_truck_last_30 = average_last_30['truck_capacity']

    average_warehouse_last_5 = average_last_5['warehouse_capacity']
    average_truck_last_5 = average_last_5['truck_capacity']

    df.set_index('date', inplace=True)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    last_sequence = scaled_data[-5:]
    predicted = []

    predict_days = 14
    for _ in range(predict_days):
        prediction = model.predict(last_sequence[np.newaxis, :, :])
        predicted.append(prediction[0])
        last_sequence = np.vstack((last_sequence[1:], prediction))

    predicted = scaler.inverse_transform(predicted)

    predicted_df = pd.DataFrame(predicted, columns=df.columns)
    predicted_df.index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=predict_days)
    predicted_df['warehouse_capacity'] = predicted_df['warehouse_capacity'].astype(int)
    predicted_df['truck_capacity'] = predicted_df['truck_capacity'].astype(int)
    predicted_df = predicted_df.reset_index()
    predicted_df.rename(columns={'index': 'date'}, inplace=True)
    warehouse_evaluation = []
    truck_evaluation = []
    for i in range(len(predicted_df)):
        warehouse_evaluation.append(evaluate(average_warehouse_last_30, average_warehouse_last_5,
                                             predicted_df.iloc[i]['warehouse_capacity']))
        truck_evaluation.append(evaluate(average_truck_last_30, average_truck_last_5,
                                         predicted_df.iloc[i]['truck_capacity']))
    predicted_df['warehouse_evaluation'] = warehouse_evaluation
    predicted_df['truck_evaluation'] = truck_evaluation
    return predicted_df


def toggle_figure():
    st.session_state.visualize = not st.session_state.visualize


def toggle_predict():
    st.session_state.predict = not st.session_state.predict


def page_1():
    # Display the DataFrame
    st.write("### Data:")
    st.dataframe(st.session_state.df)

    # 1. ADD ROW

    # Button to trigger the "modal"
    if st.button("Add row"):
        st.session_state.add_row = True

    if st.session_state.get("add_row", False):
        add_row()

    if st.button("Visualize"):
        toggle_figure()

    if st.session_state.visualize:
        fig = visualize(st.session_state.df)

        st.session_state.figure = fig

        st.write('### Visualize:')
        st.pyplot(st.session_state.figure)

        # st.rerun()

    if st.button("Predict"):
        toggle_predict()

    current_date = pd.to_datetime(st.session_state.df['date']).max()
    if st.session_state.predict:
        df_pred_lstm = predict(st.session_state.df, model_lstm, sc, cur_date=current_date)
        df_pred_rnn = predict(st.session_state.df, model_rnn, sc, cur_date=current_date)
        df_pred_gru = predict(st.session_state.df, model_gru, sc, cur_date=current_date)

        lstm_fig = visualize(st.session_state.df, df_pred_lstm)
        rnn_fig = visualize(st.session_state.df, df_pred_rnn)
        gru_fig = visualize(st.session_state.df, df_pred_gru)

        st.session_state.lstm_fig = lstm_fig
        st.session_state.rnn_fig = rnn_fig
        st.session_state.gru_fig = gru_fig

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Prediction using LSTM model")
            st.pyplot(st.session_state.lstm_fig)

        with col2:
            st.dataframe(df_pred_lstm[['date', 'warehouse_capacity', 'truck_capacity', 'warehouse_evaluation', 'truck_evaluation']])

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Prediction using RNN model")
            st.pyplot(st.session_state.rnn_fig)

        with col2:
            st.dataframe(df_pred_rnn[['date', 'warehouse_capacity', 'truck_capacity', 'warehouse_evaluation', 'truck_evaluation']])

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Prediction using GRU model")
            st.pyplot(st.session_state.gru_fig)

        with col2:
            st.dataframe(df_pred_gru[['date', 'warehouse_capacity', 'truck_capacity', 'warehouse_evaluation', 'truck_evaluation']])


def page_2():
    pass


# PAGES = {
#     "Visualization": page_1,
#     "Data": page_2
# }


def main():
    if 'df' not in st.session_state:
        # df = pd.DataFrame(columns=columns)
        # df.set_index("date", inplace=True)
        st.session_state.df = fetch_data()

    if 'page_2_df' not in st.session_state:
        st.session_state.page_2_df = pd.DataFrame(columns=columns)

    if 'figure' not in st.session_state:
        st.session_state.figure = ''

    if 'visualize' not in st.session_state:
        st.session_state.visualize = False

    if 'predict' not in st.session_state:
        st.session_state.predict = False

    # st.sidebar.title("Navigation")
    # choice = st.sidebar.selectbox("Select an option", list(PAGES.keys()))
    #
    # PAGES[choice]()
    page_1()


if __name__ == "__main__":
    main()

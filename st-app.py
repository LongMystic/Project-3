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

st.set_page_config("Warehouse Forecasting")

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


def predict(df, model, scaler: MinMaxScaler, cur_date):
    scaled_df = scaler.transform(df.drop(['id', 'date'], axis=1))
    scaled_df = scaled_df[-14:, :]

    # X = []
    # for i in range(30):
    #     X.append(scaled_df[])
    X = [scaled_df]
    X = np.array(X)

    predictions = []
    for _ in range(14):  # Predict for the next 14 days
        next_pred = model.predict(X[-30:, :, :])
        predictions.append(sc.inverse_transform(next_pred))
        # Update the sequence with the new prediction
        X = np.append([next_pred], X, axis=1)

    data = {
        'date': [],
        'price': [],
        'warehouse_capacity': [],
        'truck_capacity': []
    }

    for i in range(14):
        data['date'].append(cur_date)
        data['price'].append(int(predictions[i][0][0]))
        data['warehouse_capacity'].append(int(predictions[i][0][1]))
        data['truck_capacity'].append(int(predictions[i][0][2]))
        cur_date = cur_date + pd.Timedelta(days=1)

    return pd.DataFrame(data)


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
        df_pred_lstm = predict(st.session_state.df, model_lstm, sc , cur_date=current_date)
        df_pred_rnn = predict(st.session_state.df, model_rnn, sc,  cur_date=current_date)
        df_pred_gru = predict(st.session_state.df, model_gru, sc,  cur_date=current_date)

        lstm_fig = visualize(st.session_state.df, df_pred_lstm)
        rnn_fig = visualize(st.session_state.df, df_pred_rnn)
        gru_fig = visualize(st.session_state.df, df_pred_gru)

        st.session_state.lstm_fig = lstm_fig
        st.session_state.rnn_fig = rnn_fig
        st.session_state.gru_fig = gru_fig

        st.write("### Prediction using LSTM model")
        st.pyplot(st.session_state.lstm_fig)

        st.write("### Prediction using RNN model")
        st.pyplot(st.session_state.rnn_fig)

        st.write("### Prediction using GRU model")
        st.pyplot(st.session_state.gru_fig)


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

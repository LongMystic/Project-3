import streamlit as st
import pandas as pd
import numpy as np
import pymysql.cursors
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

# from keras.api.models import load_model

connection = pymysql.connect(
    host='localhost',
    user='root',
    password='Liquid@123',
    database='prj3',
    cursorclass=pymysql.cursors.DictCursor
)

st.set_page_config("Warehouse Forecasting")


# @st.cache_resource
# def load_my_model():
#     return load_model("final.h5")
#
#
# model = load_my_model()
# sc = joblib.load("scaler.pkl")

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
            insert_row_to_db()
            st.session_state.add_row = False
            st.rerun()  # Refresh the app to display the updated DataFrame
    with col2:
        if st.button("Cancel"):
            st.session_state.add_row = False  # Close modal without saving
            st.rerun()


def visualize(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['date'], df['warehouse_capacity'], label='warehouse_capacity', marker='o')
    ax.plot(df['date'], df['truck_capacity'], label='truck_capacity', marker='x')

    # Adding labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Capacity Over Time')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def toggle_figure():
    st.session_state.visualize = not st.session_state.visualize


def page_1():
    # Display the DataFrame
    st.write("### Data:")
    st.dataframe(st.session_state.df)

    ### 1. ADD ROW

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
        st.pyplot(fig)

        # st.rerun()


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

    # st.sidebar.title("Navigation")
    # choice = st.sidebar.selectbox("Select an option", list(PAGES.keys()))
    #
    # PAGES[choice]()
    page_1()


if __name__ == "__main__":
    main()

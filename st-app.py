import streamlit as st
import pandas as pd
import numpy as np
import pymysql.cursors
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

from keras.api.models import load_model

connection = pymysql.connect(
    host='localhost',
    user='root',
    password='juggernautlong2003',
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


def page_1():
    pass


def page_2():
    pass


PAGES = {
    "Visualization": page_1,
    "Data": page_2
}


def main():
    if 'page_1_df' not in st.session_state:
        st.session_state.page_1_df = pd.DataFrame(columns=columns)
    if 'page_2_df' not in st.session_state:
        st.session_state.page_2_df = pd.DataFrame(columns=columns)

    if 'figure' not in st.session_state:
        st.session_state.figure = ''

    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Select an option", list(PAGES.keys()))

    PAGES[choice]()


if __name__ == "__main__":
    main()

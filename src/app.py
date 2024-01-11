import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path
import joblib

main_dir = Path("../")

with st.container():
    st.title("Telco Churn Prediction  :eyes:")
    st.subheader("Predict if a customer will churn or not")


with st.container():
    st.write("----")
    st.write("""
            Input your observation here :point_down:
            """)
    columns = ['Day Mins', 'Day Calls',
       'Day Charge', 'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins',
       'Night Calls', 'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge']
    columns_left = columns[:len(columns)//2]
    columns_right = [col for col in columns if col not in columns_left]
    col1, col2 = st.columns(2)
    with col1:
        for col in columns_left:
            st.number_input(label=col, key=f"{'_'.join(col.lower().split(' '))}", min_value=-100, placeholder="Insert a number here...")
            # st.write(f"The {col} input is", num)
    with col2:
        for col in columns_right:
            st.number_input(label=col, key=f"{'_'.join(col.lower().split(' '))}", min_value=-100, placeholder="Insert a number here...")
            # st.write(f"The {col} input is", num)
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    def click_button():
        st.session_state.clicked = True
    st.button("Submit", on_click=click_button)

def add_new_record(df_current):
    model_dir = main_dir / "model"
    new_row = [st.session_state[f"{'_'.join(col.lower().split(' '))}"] for col in columns]
    df_current.loc[len(df_current)] = new_row
    stddized = joblib.load(model_dir / "stdscaler.pkl").transform(df_current)
    df_current = pd.DataFrame(stddized, columns=df_current.columns)
    df_current.loc[0, ['Churn?']] = joblib.load(model_dir / "knn_clf2.pkl").predict(df_current.loc[[0], :])
    return df_current

with st.container():
    st.write("-----")
    st.subheader("Prediction")
    st.write("Predicted result will come below :point_down:")
    df = pd.DataFrame(data=[], columns=columns)
    if st.session_state.clicked:
        df = add_new_record(df)
        st.session_state.clicked = False  # Reset clicked state
    st.dataframe(df)
    st.write("Note: the feature values are standardized once submitted into the form.")




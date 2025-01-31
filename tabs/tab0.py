# tabs/tab0.py
import streamlit as st

def show_login(books_df):
    if "history" not in st.session_state:
        st.session_state["history"] = []

    st.subheader("Login")
    user_ids = ["Guest user"] + list(books_df["User-ID"].unique())

    st.selectbox(
        "Choose a user:", 
        options=user_ids,
        key="temp_user",
        index=user_ids.index(st.session_state.user_id),
        on_change=lambda: setattr(st.session_state, 'user_id', st.session_state.temp_user)
    )
    
    st.write(f"### Welcome, **{st.session_state.user_id}**!")

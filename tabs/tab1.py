# tabs/tab1.py
import streamlit as st
from utils import recommend_book_svd, render_aligned_image

@st.fragment
def show_user_recommendations(books_df, svd_model):
    if "history" not in st.session_state:
        st.session_state["history"] = []
        
    st.subheader("📚 Find and get recommendations")
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "Guest user"
    selected_user = st.session_state.user_id
    
    if selected_user not in st.session_state:
        st.session_state[selected_user] = []
        
    if st.button("Show Recommendations", key="recommendations_svd"):
        with st.spinner("Loading recommendations..."):
            recommendations = recommend_book_svd(selected_user, books_df, svd_model)

        if recommendations:
            st.session_state[selected_user].append({
                "type": "user",
                "user": selected_user,
                "recommendations": [book_name for book_name, _, _ in recommendations],
            })
            
            st.subheader("🔍 Recommendations for you:")
            rows = [recommendations[:5], recommendations[5:10]]

            for row in rows:
                cols = st.columns(len(row))
                for idx, col in enumerate(cols):
                    book_name, rating, poster = row[idx]
                    col.markdown(render_aligned_image(poster, book_name), unsafe_allow_html=True)
        else:
            st.warning("No recommendations found for this user.")

    if st.checkbox("View recommendation history", key="history_user"):
        st.subheader("📜 History of recommendations")
        if selected_user in st.session_state:
            for entry in st.session_state[selected_user]:
                if entry.get("type") == "user":
                    st.write(f"**User : {entry['user']}** → {', '.join(entry['recommendations'])}")
        else:
            st.write("No history found for this user.")
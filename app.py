import os
import streamlit as st
from src.utils.helper import get_username

if 'username' not in st.session_state:
    st.session_state.username = get_username()

if st.session_state.username == os.getenv('admin'):
    pages =  [
                st.Page("pages/home.py", title="ChatApp", icon=':material/chat:'),
            ]
else:
    pages =  [
    	      st.Page("pages/home.py", title="ChatApp", icon=':material/chat:')
            ]

pg = st.navigation(pages, position='sidebar')
pg.run()

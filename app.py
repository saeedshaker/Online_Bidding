#app.py
import Main
import InfoPage
import Example
import TryItOut
import streamlit as st
PAGES = {
    "Home page": Main,
    "Demo": Example,
    "Try It Out": TryItOut,
    "Technical Details": InfoPage
}
#    "Try It Out More": TryItOutMore,
st.sidebar.title('App by: Saeed Shaker')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
st.sidebar.write('\n')
page = PAGES[selection]
#if page == TryItOut:
#    page.app({'upload': 0, 'confirm1': 0, 'conf1_state': {}, 'confirm2': 0, 'conf2_state': {} })
#elif page == TryItOutMore:
#    page.app(state_dict)
#else:
#    page.app()
page.app()
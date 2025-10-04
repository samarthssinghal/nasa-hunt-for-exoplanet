import streamlit as st
from components.header import render_header
from components.navigation import render_navigation
from pages import home, search

st.set_page_config(
    page_title="NASA Exoplanet Detection Lab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    render_header()

    render_navigation()


if __name__ == "__main__":
    main()

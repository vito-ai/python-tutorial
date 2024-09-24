import streamlit as st

from utils import display_result, page_setup

RTZR_LOGO_URL = "https://www.rtzr.ai/rtzr_logo.svg"
RTZR_HOMEPAGE_URL = "http://rtzr.ai"
API_TUTORIAL_URL = "https://developers.rtzr.ai/docs/authentications"
AUDIO_FILE_PATH = "./resource"

# streamlit setting

st.set_page_config(layout="wide", page_title="STT and Summary", page_icon=RTZR_LOGO_URL)

if __name__ == "__main__":
    # streamlit main
    with st.container():
        my_upload: st.file_uploader = page_setup(
            RTZR_LOGO_URL, RTZR_HOMEPAGE_URL, API_TUTORIAL_URL
        )
        display_result(AUDIO_FILE_PATH, my_upload)

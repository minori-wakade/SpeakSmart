import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CNN_modified.test_emotion import run_streamlit_app

run_streamlit_app()

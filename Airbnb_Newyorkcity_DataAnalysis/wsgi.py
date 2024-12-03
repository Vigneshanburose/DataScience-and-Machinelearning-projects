import os
import subprocess

if __name__ == "__main__":
    # Start the Streamlit app as a subprocess
    streamlit_script = "app.py"  
    os.system(f"streamlit run {streamlit_script}")

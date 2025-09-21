import subprocess
import sys

if __name__ == '__main__':
    sys.path.append('/src')
    subprocess.run(["streamlit", "run", "app/app.py"])

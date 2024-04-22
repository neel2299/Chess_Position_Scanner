from pyngrok import ngrok
from threading import Thread
import os


ngrok.set_auth_token('2fR9ugzvoRsFUwxVHq6Qt9YJtv8_DaMJXPdJjauYwuXnfBzp')

def run_streamlit():
    # Change the port if 8501 is already in use or if you prefer another port
    os.system('python3.6 -m streamlit run trial.py --server.port 8501')

thread = Thread(target=run_streamlit)
thread.start()
public_url = ngrok.connect(addr='8501', proto='http', bind_tls=True)
print('Your Streamlit app is live at:', public_url)

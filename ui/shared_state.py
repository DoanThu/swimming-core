import streamlit as st
import threading
import time
from ui.socket_receiver import SocketReceiver
from config.general import SERVER_HOST, SERVER_PORT

class SharedMetrics:
    def __init__(self):
        self.athlete_data = {}
        self.last_updated = 0

@st.cache_resource
def get_shared_metrics():
    return SharedMetrics()

class StreamManager:
    def __init__(self):
        self.receiver = SocketReceiver(SERVER_HOST, SERVER_PORT)
        self.latest_data = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.running: return True
        result = self.receiver.start()
        if result is True:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
        return result

    def _update_loop(self):
        while self.running:
            data = self.receiver.get_latest()
            if data:
                with self.lock:
                    self.latest_data = data
            else:
                time.sleep(0.001)
        self.running = False

    def get_latest_frame(self):
        with self.lock:
            return self.latest_data
    
    def stop(self):
        self.running = False
        self.receiver.stop()

@st.cache_resource
def get_stream_manager():
    return StreamManager()

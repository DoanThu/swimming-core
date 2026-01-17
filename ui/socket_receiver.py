import socket
import threading
import queue
from ui.utils import receive_data

class SocketReceiver:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.sock = None
        self.is_running = False

    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.is_running = True
            t = threading.Thread(target=self._receive_loop, daemon=True)
            t.start()
            return True
        except Exception as e:
            return str(e)

    def _receive_loop(self):
        while self.is_running and not self.stop_event.is_set():
            try:
                data = receive_data(self.sock)
                if data is None: break
                if self.queue.full():
                    try: self.queue.get_nowait()
                    except queue.Empty: pass
                self.queue.put(data)
            except Exception:
                break
        self.is_running = False
        if self.sock: 
            try: self.sock.close()
            except: pass

    def get_latest(self):
        try: return self.queue.get_nowait()
        except queue.Empty: return None
    
    def stop(self):
        self.stop_event.set()
        self.is_running = False
        if self.sock: 
            try: self.sock.close()
            except: pass

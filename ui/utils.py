import struct
import pickle

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def receive_data(sock):
    """
    Receive pickled data from the socket server.
    Expected format:
    1. Payload size (unsigned long)
    2. Number of items in the list (pickled)
    3. For each item:
        a. Item size (unsigned long)
        b. Item data (pickled)
    """
    # 1. Get number of items (pickled)
    payload_size = struct.calcsize("L")
    packed_msg_size = recvall(sock, payload_size)
    if not packed_msg_size: return None
    msg_size = struct.unpack("L", packed_msg_size)[0]
    
    data = recvall(sock, msg_size)
    if not data: return None
    num_items = pickle.loads(data)
    
    items = []
    for _ in range(num_items):
        packed_msg_size = recvall(sock, payload_size)
        if not packed_msg_size: return None
        msg_size = struct.unpack("L", packed_msg_size)[0]
        
        data = recvall(sock, msg_size)
        if not data: return None
        items.append(pickle.loads(data))
    return items

CSS_STYLE = """
    <style>
    /* Hide Streamlit Default Header and Footer */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 99% !important;
    }
    h1, h2, h3 { margin-top: 0rem !important; padding-top: 0rem !important; }
    h1 { font-size: 1.8rem; font-weight: 700; }
    
    /* Coach Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .big-metric {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    .small-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .stroke-badge {
        background-color: #e0e7ff;
        color: #4338ca;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """
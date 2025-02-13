import socket,cv2,pickle, struct
from config.general import FPS_RATE, VIDEO_PATH, SAVE_AFTER_SECONDS
from main_process.core_process import MainCalculation


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
port = 9999
serversocket.bind((host, port))
serversocket.listen(5)

print("Server started")


frame_idx = 0
try: 
    while True:
        clientsocket, addr = serversocket.accept()
        try:
            if clientsocket:
                print("Got a connection from %s" % str(addr))
                # cap = cv2.VideoCapture(VIDEO_PATH)
                cap = cv2.VideoCapture(0)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                calculate_frame = MainCalculation(fps, lane_type='detection')
                frame_idx = -1
                while(cap.isOpened()):
                    frame_idx += 1

                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)

                        if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                            print(f'>>>>> {SAVE_AFTER_SECONDS} seconds elapsed. Current frame idx is {frame_idx}')

                        if frame_idx % FPS_RATE != 0: 
                            continue
                        annotated_frame, frame_data, updated_speed = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=False)

                        a = pickle.dumps(annotated_frame)
                        message = struct.pack("L", len(a))+a

                        a = pickle.dumps(frame)
                        message += struct.pack("L", len(a))+a
                        clientsocket.sendall(message)

                    else:
                        print('Video ended')
                        break
        except Exception as e:
            print("Closed a connection from %s" % str(addr))
            clientsocket.close()
except KeyboardInterrupt:
    serversocket.close()
    print('Server closed')
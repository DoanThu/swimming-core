import argparse
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.video_process import run_video
from main_process.streaming_process import run_stream
from main_process.socket_process import run_socket
from config.general import RESOLUTION


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, help="mode can be VIDEO/STREAMING/SOCKET", required=True)
    args = vars(ap.parse_args())

    logging.info(f'RUNNING IN {args['mode']} MODE ' + '---'*5)
    if args['mode'] == 'VIDEO':
        run_video(debug=True, save_json=False, save_csv=True, fx=0.5, fy=0.5, lane_type='segmentation')
    elif args['mode'] == 'STREAMING':
        run_stream(debug=False, save_json=False, save_csv=True, fx=0.5, fy=0.5, lane_type='detection')
    elif args['mode'] == 'SOCKET':
        run_socket(debug=False, save_json=False, save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1], lane_type='detection')
    
   
import argparse
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.video_process import run_video
from main_process.socket_process import run_socket
from config.general import RESOLUTION, VIDEO_PATH
import os


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, help="mode can be VIDEO/SOCKET", required=True)
    args = vars(ap.parse_args())

    logging.info(f"RUNNING IN {args['mode']} MODE " + '---'*5)
    if args['mode'] == 'VIDEO':
        path = 'videos/2025/2025Mar22_resized'
        for file in os.listdir(path):
            if file[-3:] != 'MP4': continue 
            if file[0] == '.': continue
            full_path = f'{path}/{file}'
            run_video(debug=True, video_path=full_path, save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
    elif args['mode'] == 'SOCKET':
        run_socket(debug=False, save_json=False, save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
    
   
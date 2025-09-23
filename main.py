import argparse
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.video_process import run_video, run_video_multi
from main_process.socket_process import run_socket
from config.general import RESOLUTION


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, help="mode can be VIDEO_SINGLE/VIDEO_MULTI/SOCKET", required=True)
    ap.add_argument("--debug", type=bool, help="show metrics on screen", default=False)
    args = vars(ap.parse_args())

    logging.info(f"RUNNING IN {args['mode']} MODE " + '---'*5)
    if args['mode'] == 'VIDEO_SINGLE':
        run_video(debug=args['debug'], save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
    elif args['mode'] == 'VIDEO_MULTI':
        run_video_multi(debug=args['debug'], save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
    elif args['mode'] == 'SOCKET':
        run_socket(debug=False, save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
    
   
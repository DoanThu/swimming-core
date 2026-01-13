import argparse
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.video_process import run_video, run_video_multi, run_video_multi_threaded
from main_process.socket_process import run_socket
from config.general import RESOLUTION


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, help="mode can be VIDEO_SINGLE/VIDEO_MULTI/SOCKET", required=True)
    ap.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="show metrics on screen")
    args = vars(ap.parse_args())

    logging.info(f"RUNNING IN {args['mode']} MODE " + '---'*5)
    # This mode will track only the swimmer in the middle of the screen
    if args['mode'] == 'VIDEO_SINGLE':
        run_video(debug=args['debug'], save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
    # This mode track all swimmers on the screen
    elif args['mode'] == 'VIDEO_MULTI':
        # plan to deprecate this function
        # this function is slower than the threaded version
        # run_video_multi(debug=args['debug'], save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
        run_video_multi_threaded(debug=args['debug'], save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
    elif args['mode'] == 'SOCKET':
        run_socket(debug=False, save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
    else:
        logging.error('NO MODE MATCHED!')
    
   
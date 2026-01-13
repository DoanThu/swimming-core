import argparse
import logging 
import os
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.video_process import run_video, run_video_multi, run_video_multi_threaded
from main_process.socket_process import run_socket
from config.general import RESOLUTION, VIDEO_PATH


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
    ap.add_argument("--source", type=str, help="path to video file or directory", default=VIDEO_PATH)
    ap.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="show metrics on screen")
    args = vars(ap.parse_args())

    logging.info(f"RUNNING IN {args['mode']} MODE " + '---'*5)

    video_paths = []
    if os.path.isdir(args['source']):
        for f in os.listdir(args['source']):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(args['source'], f))
    else:
        video_paths.append(args['source'])

    for video_path in video_paths:
        logging.info(f"Processing video: {video_path}")
    # This mode will track only the swimmer in the middle of the screen
        if args['mode'] == 'VIDEO_SINGLE':
            run_video(video_path, debug=args['debug'], save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
        # This mode track all swimmers on the screen
        elif args['mode'] == 'VIDEO_MULTI':
            # plan to deprecate this function
            # this function is slower than the threaded version
            # run_video_multi(video_path, debug=args['debug'], save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
            run_video_multi_threaded(video_path, debug=args['debug'], save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
        elif args['mode'] == 'SOCKET':
            run_socket(debug=False, save_csv=True, fx=RESOLUTION[0], fy=RESOLUTION[1])
        else:
            logging.error('NO MODE MATCHED!')
    
   
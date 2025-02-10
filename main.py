from config.general import MODE
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.video_process import run_video
from main_process.streaming_process import run_stream


if __name__ == '__main__':
    logging.info(f'RUNNING IN {MODE} MODE' + '---'*5)
    if MODE == 'VIDEO':
        run_video(debug=True, save_json=False, save_csv=True, fx=0.5, fy=0.5)
    elif MODE == 'STREAMING':
        run_stream(debug=False, save_json=False, save_csv=True, fx=0.5, fy=0.5)
    
    
   
from config.general import MODE
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
from main_process.video_process import run_video
from main_process.streaming_process import run_stream


if __name__ == '__main__':
    if MODE == 'VIDEO':
        run_video(debug=True, save_json=False, save_csv=True, fx=1.0, fy=1.0)
    elif MODE == 'STREAMING':
        run_stream(debug=False, save_json=False, save_csv=True, fx=0.5, fy=0.5)
    
    
   
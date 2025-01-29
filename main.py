from config.general import MODE, SAVE_AFTER_SECONDS, SAVE_JSON_PATH, SAVE_CSV_PATH, VIDEO_PATH, DEVICE_ID, SAVE_VIDEO_PATH
import cv2 
from utils.file_utils import write_to_csv
from postprocess.data import FrameData
from typing import List
from main_process.main_processing import MainCalculation
import os
import sys
import logging 
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

frame_data_list: List[FrameData] = []

def run_video(debug=False, save_json=False, save_csv=False, out_video=SAVE_VIDEO_PATH, fx=1, fy=1):
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    if frame_width == 0 and frame_height == 0:
        logging.info(f'Cannot find video at {VIDEO_PATH}.')
        return 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = 0
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps, (frame_width, frame_height))
    
    logging.info(f'frame_width={frame_width}, frame_height={frame_width}, fps = {fps}')

    if save_csv:
        if os.path.exists(SAVE_CSV_PATH):
            os.remove(SAVE_CSV_PATH)
    
    calculate_frame = MainCalculation(fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out_frame, frame_data = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug)
            frame_idx += 1
            
            # save SAVE_AFTER_SECONDS frames in json format
            if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                if save_json:
                    filename = SAVE_JSON_PATH.format(frame_idx//(SAVE_AFTER_SECONDS*fps))
                    save_json([_frame.__dict__ for _frame in frame_data_list[frame_idx-(SAVE_AFTER_SECONDS*fps):]], filename)
                    logging.info(f'Saved json file to {filename}')
                else:
                    logging.debug('>>>>> {} seconds elapsed'.format(SAVE_AFTER_SECONDS))
                    
            if save_csv:
                data = str(frame_data.red_marker) + ',' + str(frame_data.direction) + ',' + str(frame_data.speed) + ',' + ','.join(str(p) for p in frame_data.skeleton)
                write_to_csv(data, SAVE_CSV_PATH)
            
            if out_video:
                output.write(out_frame)
            
        else:
            break
        
    cap.release()
    output.release()
    
    if debug:
        logging.debug(f'Annotated video is saved at {out_video}')
    if save_csv:
        logging.debug(f'Csv file is saved at {SAVE_CSV_PATH}')


def run_stream(debug=False, save_json=False, save_csv=False, out_video=SAVE_VIDEO_PATH, fx=1, fy=1):
    cap = cv2.VideoCapture(DEVICE_ID)
    frame_width, frame_height = int(cap.get(3)*fx), int(cap.get(4)*fy)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_idx = 0
    output = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'),
                             fps, (frame_width, frame_height))
    
    logging.info(f'frame_width={frame_width}, frame_height={frame_width}, fps = {fps}')

    if save_csv:
        if os.path.exists(SAVE_CSV_PATH):
            os.remove(SAVE_CSV_PATH)
    
    calculate_frame = MainCalculation(fps)
    
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret:
                out_frame, frame_data = calculate_frame.swimming_calculation(frame=frame, frame_idx=frame_idx, debug=debug)
                frame_idx += 1
                
                # save SAVE_AFTER_SECONDS frames in json format
                if frame_idx % (SAVE_AFTER_SECONDS*fps) == 0:
                    if save_json:
                        filename = SAVE_JSON_PATH.format(frame_idx//(SAVE_AFTER_SECONDS*fps))
                        save_json([_frame.__dict__ for _frame in frame_data_list[frame_idx-(SAVE_AFTER_SECONDS*fps):]], filename)
                        logging.info(f'Saved json file to {filename}')
                    else:
                        logging.debug('>>>>> {} seconds elapsed'.format(SAVE_AFTER_SECONDS))
                        
                if save_csv:
                    data = str(frame_data.red_marker) + ',' + str(frame_data.direction) + ',' + str(frame_data.speed) + ',' + ','.join(str(p) for p in frame_data.skeleton)
                    write_to_csv(data, SAVE_CSV_PATH)

                
                if out_video:
                    output.write(out_frame)
                
            else:
                if save_csv:
                    logging.info(f'Saved csv file to {SAVE_CSV_PATH}')

                cap.release()
                output.release()
        except KeyboardInterrupt:
            cap.release()
            output.release()
            if save_csv:
                logging.info(f'Saved csv file to {SAVE_CSV_PATH}')
            logging.info(f'Saved video to {out_video}')
            sys.exit()
    
    if debug:
        logging.debug(f'Annotated video is saved at {out_video}')
    if save_csv:
        logging.debug(f'Csv file is saved at {SAVE_CSV_PATH}')



if __name__ == '__main__':
    if MODE == 'VIDEO':
        run_video(debug=False, save_json=False, save_csv=True)
    elif MODE == 'STREAMING':
        run_stream(debug=False, save_json=False, save_csv=True)
    
    
   
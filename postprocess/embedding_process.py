from postprocess.data import FrameDataConst 
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class EmbeddingProcess:
    def __init__(self, window_size=100):
        self.hip_length = 0
        self.torso_length = 0
        self.embedding_list = []
        self.window_size = window_size

        # Create options for Image Embedder
        base_options = python.BaseOptions(model_asset_path='./models/mobilenet_v3_small.tflite')
        l2_normalize = True #@param {type:"boolean"}
        quantize = True #@param {type:"boolean"}
        self.options = vision.ImageEmbedderOptions(base_options=base_options, 
                                                   l2_normalize=l2_normalize, 
                                                   quantize=quantize)

    def get_embeddings(self, frame, frame_data):
        skeleton = frame_data.skeleton
        lhip = skeleton[0][11].cpu().numpy().astype(np.int32)
        rhip = skeleton[0][12].cpu().numpy().astype(np.int32)
        nose = skeleton[0][0].cpu().numpy().astype(np.int32)
        width, height = frame.shape[1], frame.shape[0]
        if frame_data.frame_orientation == FrameDataConst.HORIZONTAL:
            if self.hip_length == 0:
                self.hip_length = abs(lhip[1]-rhip[1])
            if self.torso_length == 0:
                self.torso_length = self.hip_length*3

            cut_frame = np.array([])
            if frame_data.direction == FrameDataConst.RIGHT:
                cut_frame = frame[lhip[1]:min(height,lhip[1]+self.hip_length), max(0,lhip[0]-self.torso_length):lhip[0]].copy()
            elif frame_data.direction == FrameDataConst.LEFT:
                cut_frame = frame[rhip[1]:min(height,rhip[1]+self.hip_length), lhip[0]:min(width,lhip[0]+self.torso_length)].copy()

            cv2.imwrite('cut_frame.jpg', cut_frame)
            if cut_frame.shape[0] == 0:
                return

            # Create Image Embedder
            with vision.ImageEmbedder.create_from_options(self.options) as embedder:
                # Format images for MediaPipe
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cut_frame)

                embedding = embedder.embed(mp_img)
                self.embedding_list.append(embedding)
                if len(self.embedding_list) > self.window_size:
                    self.embedding_list.pop(0)

                similarity_list = []
                for e in self.embedding_list[:-1]:
                  similarity = vision.ImageEmbedder.cosine_similarity(embedding.embeddings[0], e.embeddings[0])
                  similarity_list.append(similarity)
                if len(similarity_list) >= 3:
                    argsort = np.argsort(similarity_list)
                    print(len(similarity_list) - argsort[-1], similarity_list[argsort[-1]])
                    print(len(similarity_list) - argsort[-2], similarity_list[argsort[-2]]) 
                    print(len(similarity_list) - argsort[-3], similarity_list[argsort[-3]]) 
                    print('>>>>>'*5)

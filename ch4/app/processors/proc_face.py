"""Face processor utilities"""
from os.path import join
import logging
import collections
import operator
import json

import cv2 as cv
import face_recognition as fr
import numpy as np

from megapixels.models.bbox import BBox
from app.settings import app_cfg as cfg
from megapixels.utils import im_utils

class FaceUtils:

  def __init__(self):
    pass



class FaceRecognizer:

  def __init__(self):
    self.encodings = []
    self.num_jitters = cfg.FACEREC_NUM_JITTERS
    self.log = logging.getLogger()
    
  def set_encodings(self, fp_encodings):
    """Sets the dataset to match to"""
    with open(fp_encodings, 'r') as fp:
      encodings = json.load(fp)
    self.encodings = collections.OrderedDict(sorted(encodings.items()))  
    self.encodings_flat = [v for k, v in self.encodings.items()]
    self.encodings_idx = {i: k for i, k in enumerate(self.encodings)}

  def match(self, fp_query, num_matches=3):
    """Returns best match as images"""
    if not self.encodings:
      self.log.error('Encodings were not set')
      return None
    im_query = cv.imread(fp_query)

    im_query = im_utils.resize(im_query,
      width=cfg.QUERY_IMAGE_WIDTH, height=cfg.QUERY_IMAGE_HEIGHT)
    enc_query = self.encode(im_query)
    if enc_query is None:
      self.log.error('could not encode face. probably no face detected')
      return None

    # find best match
    distances = fr.face_distance(self.encodings_flat, enc_query)
    top_idxs = np.argpartition(distances, num_matches)[0:num_matches]

    matches = []
    for i in list((range(num_matches))):
      idx = top_idxs[i]
      match_info = self.encodings_idx[idx]
      match_name, fp_im = match_info.split(':') 
      matches.append({
        'filename': fp_im,
        'person': match_name,
        'score': distances[idx]
        })

    matches = sorted(matches, key=operator.itemgetter("score"))
    return matches


  def encode(self, im):
    try:
      enc = fr.face_encodings(im, num_jitters=self.num_jitters)[0]
    except Exception as ex:
      self.log.error('No face detected. Try another image.')
      enc = None
    return enc



class HaarDetector:

  def __init__(self, cascade_name='haarcascade_frontalface_default'):
    self._cascade_name = cascade_name
    fp_cascade = join(cv.data.haarcascades, '{}.xml'.format(cascade_name))
    self._classifier = cv.CascadeClassifier(fp_cascade)

  def detect(self, im, scale=1.1, overlaps=3, flags=0, 
    min_size=60, max_size=600):
    """Detects faces with haarcascade and returns rects as list"""
    # Convert to grayscale for speedup
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # Run detector
    matches = self._classifier.detectMultiScale(im_gray, scale, overlaps, 
      flags, (min_size, min_size), (max_size, max_size))
    # Change x, y, w, h --> x1, y1, x2, y2
    return [BBox.from_haar(m) for m in matches]



# class FaceProcessor:
  
#   def __init__(self):
#     self._dlib_predictor = None

#     pass
  
#   # Define a function to detect faces using OpenCV's haarcascades
#   def detect_haarface(self, src, cascade_name=None,
#       scale_factor=1.1, overlaps=3, min_size=70, max_size=700, flags=0):
    
#     # haarcascade_frontalface_alt2
#     # haarcascade_frontalface_alt_tree
#     # haarcascade_frontalface_alt
#     # haarcascade_profileface
#     # haarcascade_frontalface_default
#     if self._classifier is None or self._cascade_name != cascade_name:
#       self._classifier = cv2.CascadeClassifier(fp_cascade)

    

#     if classifier is None:
#       # init classifier

    

#   def get_pose(self,im,landmarks):
#     """
#     Given an image and list of landmarks, calcualte the pose
#     :param im: a Numpy image in grayscale
#     :param roi: a tuple with (x1,y1,x2,y2) format
#     :return: success, rotation vector, translation vector, camera matrix, pose points
#     """
#     pose_points_idx = (30,8,36,45,48,54)

#     # 3D model points.
#     model_points = np.array([
#         (0.0, 0.0, 0.0),             # Nose tip
#         (0.0, -330.0, -65.0),        # Chin
#         (-225.0, 170.0, -135.0),     # Left eye left corner
#         (225.0, 170.0, -135.0),      # Right eye right corne
#         (-150.0, -150.0, -125.0),    # Left Mouth corner
#         (150.0, -150.0, -125.0)      # Right mouth corner
#     ])

#     size = im.shape
#     # Camera internals
#     focal_length = size[1]
#     center = (size[1]/2, size[0]/2)
#     camera_matrix = np.array(
#       [[focal_length, 0, center[0]],
#       [0, focal_length, center[1]],
#       [0, 1, 1]], dtype = "double"
#     )

#     pose_points = []
#     for j,pidx in enumerate(pose_points_idx):
#       ff = landmarks[pidx]
#       pose_points.append((ff[0],ff[1]))

#     pose_points = np.array(pose_points, dtype='double')

#     dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
#     (success, rot_vec, tran_vec) = cv2.solvePnP(
#     model_points, pose_points, 
#     camera_matrix, dist_coeffs, 
#     flags=cv2.CV_ITERATIVE)
#     # CV_P3P, CV_EPNP
#     return (success, rot_vec, tran_vec,camera_matrix,pose_points)
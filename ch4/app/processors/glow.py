import os
from os.path import join
import sys
from pathlib import Path
from glob import glob
from time import time

import numpy as np
from imageio import  mimwrite, get_writer
from PIL import Image
import cv2 as cv
import dlib
import imutils
from imutils.face_utils import FaceAligner

import megapixels.settings.global_cfg as gcfg

# glow
dir_glow = join(str(Path(os.path.abspath(__file__)).parent.parent.parent), '3rdparty')
sys.path.append(dir_glow)
import glow.demo.model as glow_utils


class Glow:

  def __init__(self):
    self.detector = dlib.get_frontal_face_detector()
    self.predictor = dlib.shape_predictor(gcfg.FP_DLIB_PREDICTOR)
    self.fa = FaceAligner(self.predictor, desiredFaceWidth=256,
      desiredLeftEye=(0.371, 0.480))


  # Input: numpy array for image with RGB channels
  # Output: (numpy array, face_found)
  def align_face(self, im, expand):

    # detect faces in the grayscale image
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    rects = self.detector(gray, 2)

    if len(rects) > 0:
      # align the face using facial landmarks
      x1, y1 = rects[0].tl_corner().x - expand, rects[0].tl_corner().y - expand
      x2, y2 = rects[0].br_corner().x + expand, rects[0].br_corner().y + expand
      expanded = dlib.rectangle(x1, y1, x2, y2)
      im_align = self.fa.align(im, gray, expanded)[:, :, ::-1]
      return im_align
    else:
      # No face found
      return None


  def generate(self, fp_a, fp_b, expand, points=3):

    # mix images
    im_a = cv.imread(fp_a)
    im_b = cv.imread(fp_b)
    im_a = imutils.resize(im_a, width=gcfg.SIZE_DLIB_DETECT)
    im_b = imutils.resize(im_b, width=gcfg.SIZE_DLIB_DETECT)
    im_a_face = self.align_face(im_a, expand=0)
    im_b_face = self.align_face(im_b, expand=0)
    im_a_face = cv.cvtColor(im_a_face, cv.COLOR_BGR2RGB)
    im_b_face = cv.cvtColor(im_b_face, cv.COLOR_BGR2RGB)
    z1 = glow_utils.encode(im_a_face)
    z2 = glow_utils.encode(im_b_face)
    ims, _ = glow_utils.mix_range(z1, z2, points)

    # save midpoint
    if not points % 2:
      print('[!] Warning: {} is even and does not have a midpoint'.format(points))
    return ims



  def resize(self, arr, res, ratio=1.):
    shape = (int(res*ratio),res)
    return np.array(Image.fromarray(arr).resize(shape, resample=Image.ANTIALIAS))

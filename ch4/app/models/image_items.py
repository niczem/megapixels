import os
from os.path import join
from pathlib import Path
import logging

import cv2 as cv
import numpy as np
import imutils

from megapixels.utils import im_utils
from megapixels.settings import global_cfg as gcfg
from app.settings import app_cfg as cfg

class ImageItem(object):

  def __init__(self, fp_src):
    self._fp_src = fp_src
    self._log = logging.getLogger()
    try:
      im = cv.imread(fp_src)
      if im is None \
        or not(im.shape[::-1][1:] >= cfg.MIN_IMAGE_SIZE) \
        or len(im.shape) == 2:
          self._rejected = True
      else:
        # resize to maximum size
        self._im = im_utils.resize(im, width=cfg.MAX_IMAGE_SIZE[0], height=cfg.MAX_IMAGE_SIZE[1])
        if self._im.shape[2] > 3:
          # remove alpha channel
          self._im = cv.cvtColor(im, cv.COLOR_RGBA2RGB)
        self._rejected = False
    except Exception as ex:
      self._log.error(ex)
      self._rejected = True
      self._im = None

  def reject(self, msg=None):
    self._log.info('rejected: {}'.format(Path(self._fp_src).name))
    self._log.info('  because: {}'.format(msg))
    self._rejected = True

  def approve(self, msg=None):
    self._log.info('approved: {}'.format(Path(self._fp_src).name))
    self._log.info('  because: {}'.format(msg))
    self._rejected  = False

  @property
  def fp_src(self):
    return self._fp_src
  
  @property
  def im(self):
    return self._im

  @property
  def rejected(self):
    return self._rejected

  @rejected.setter
  def rejected(self, val):
    self._rejected = val
  
  
  


class ImageFilterItem(ImageItem):

  def __init__(self, fp_src):
    super().__init__(fp_src)
    self._phash = None
    self._encoding = None

  @property
  def encoding(self):
    return self._encoding
  
  @encoding.setter
  def encoding(self, val):
    self._encoding = val

  @property
  def phash(self):
    return self._phash
  
  @phash.setter
  def phash(self, val):
    self._phash = val


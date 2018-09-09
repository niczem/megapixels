"""
Dataset utilities
"""
import sys
import os
from os.path import join
from glob import glob
from pathlib import Path
import shutil
import json
import csv
import pickle
import time
import logging
import itertools

import hashlib
import click
from tqdm import tqdm
import cv2 as cv
from PIL import Image
import face_recognition
import numpy as np

from megapixels.utils import im_utils, file_utils
from megapixels.settings import global_cfg as gcfg
from megapixels.models.bbox import BBox

from app.processors.proc_face import HaarDetector
from app.models.image_items import ImageFilterItem
from app.settings import app_cfg as cfg

haar_detector = HaarDetector()
face_dupe_threshold = 0.1

log = logging.getLogger()


def generate_encodings(dirs):
  encodings = {}
  log.info('generating face encodings...')
  for dir_identity in tqdm(dirs):
    dir_name = Path(dir_identity).name
    im_list = file_utils.glob_images(join(dir_identity, cfg.DIRNAME_APPROVED))
    for fp_im in im_list:
      im_name = Path(fp_im).stem
      im_key = '{}:{}'.format(dir_name, im_name)
      im = cv.imread(fp_im)
      try:
        enc = face_recognition.face_encodings(im, num_jitters=3)[0]
        encodings[im_key] = list(enc)
      except Exception as ex:
        log.error(ex)
        # don't include this image if no encoding
        log.error(' no encodings for: {}, {}'.format(dir_name, im_name))
        continue
  return encodings


def rejected(image_items):
  return [x for x in image_items if x.rejected]

def approved(image_items):
  return [x for x in image_items if not x.rejected]

def filter_grayscale(image_items, threshold=5):
  """Remove images with grayscale colormap"""
  for item in image_items:
    if item.rejected:
      continue
    b = item.im[:,:,0]
    g = item.im[:,:,1]
    mean = np.mean(np.abs(g - b))
    if mean < threshold:
      item.reject('grayscale')
  return image_items

def filter_imagehash(image_items, width=100, threshold=10):
  """Deduplicates file list of images
  :param files: (list) of items
  :returns: (list) of updated items
  """

  # iterate and update item rejected status
  for item in image_items:
    if item.rejected:
      continue
    im_sm = im_utils.resize(item.im, width=width)
    phash = im_utils.phash(im_sm)
    item.phash = phash

  for item in image_items:
    if item.rejected:
      continue  
    # Compare hash against all non-rejected hashes
    for item2 in image_items:
      if item2.rejected or item2.fp_src == item.fp_src:
        continue
      if abs(item.phash - item2.phash) < threshold:
        item2.reject('Found duplicate {} and rejected'.format(Path(item2.fp_src).name))
        break

  return image_items

def get_encoding(im):
  """Use default face_recognition methods"""
  pass

def filter_multiface(image_items, scale=1.1, width=600, dim=(cfg.MIN_FACE_SIZE, cfg.MAX_FACE_SIZE)):
  """Reject images containing more than one face"""

  for item in image_items:
    if item.rejected:
      continue
    im_resized = im_utils.resize(item.im, width=width)
    # use haar
    bboxes = haar_detector.detect(im_resized, scale=cfg.HAAR_SCALE, 
      overlaps=cfg.HAAR_OVERLAPS, min_size=dim[0], max_size=dim[1])
    if len(bboxes) > 1:
      item.reject('too many haar faces ({})'.format(len(bboxes)))
    elif len(bboxes) == 0:
      item.reject('no haar faces found')
    elif bboxes[0].width < cfg.MIN_FACE_SIZE or bboxes[0].width > cfg.MAX_FACE_SIZE:
      item.reject('haar face too small/large: {}'.format(bboxes[0].width))
    # use hog
    face_locations = face_recognition.face_locations(im_resized)
    bboxes = [BBox.from_css(x) for x in face_locations]
    if len(bboxes) > 1:
      item.reject('too many dlib faces ({})'.format(len(bboxes)))
    elif len(bboxes) == 0:
      item.reject('no dlib faces found')
    elif bboxes[0].width < cfg.MIN_FACE_SIZE or bboxes[0].width > cfg.MAX_FACE_SIZE:
      item.reject('dlib face too small/large: {}'.format(bboxes[0].width))


  return image_items

def filter_identity(image_items, dir_in, confirm=False, width=600, ext='jpg'):
  """Filters out images that are not the same person as ground truth
  """

  # first, generate encodings for all image items
  for item in image_items:
    if item.rejected:
      continue
    try:
      im_resized = im_utils.resize(item.im, width=width)
      enc_item = face_recognition.face_encodings(im_resized, num_jitters=3)[0]
      item.encoding = enc_item
    except:
      item.reject('Does not contain face/encoding')
    # iterate and compare

  if confirm:
    # ensure ground truth image exists
    dir_ground_truth = join(dir_in, cfg.DIRNAME_GROUND_TRUTH)
    glob_list = file_utils.glob_images(dir_ground_truth)
    if not len(glob_list) > 0:
      log.info('*******************************')
      log.error("Error: there must be at least one ground truth image")
      log.error('Add a confirmed face image in the folder: {}'.format(dir_ground_truth))
      log.info('*******************************')
      return image_items
    else:
      fp_truth = glob_list[0]

    # try loading ground truth image
    im_truth = cv.imread(fp_truth)
    if im_truth is None \
        or not(im_truth.shape[::-1][1:] >= cfg.MIN_IMAGE_SIZE) \
        or len(im_truth.shape) == 2:
          log.info('*******************************')
          log.error("Error: could not open confirmation image or too small")
          log.info('*******************************')
          return image_items

    # try getting ground truth image encoding
    try:
      im_truth_resized = im_utils.resize(im_truth, width=width)
      enc_truth = face_recognition.face_encodings(im_truth_resized, num_jitters=3)[0]
    except Exception as ex:
      log.error(ex)
      log.info('*******************************')
      log.error('The ground truth image does not contain a face')
      log.info('*******************************')
      return image_items

    # then compare ground truth to all other faces
    for item in image_items:
      if item.rejected:
        continue
      face_delta = np.linalg.norm([item.encoding] - enc_truth, axis=1)[0]
      if face_delta > cfg.FACE_GROUND_TRUTH_THRESH:
        item.reject("Does not match target face")
      else:
        log.info('face delta: {}'.format(face_delta))

  # finally, match all faces against each other to reject duplicates
  for item in image_items:
    if item.rejected:
      continue
    for item2 in image_items:
      if item2.rejected or item.fp_src == item.fp_src:
        continue
      log.info('enc: {}'.format(item.encoding))
      log.info('enc2: {}'.format(item2.encoding))
      face_delta = np.linalg.norm([item2.encoding] - item.encoding, axis=1)[0]
      log.info('delta: {}'.format(face_delta))
      if face_delta < cfg.FACE_DUPLICATE_THRESH:
        item2.reject('duplicate face/image. thresh: {}'.format(face_delta))
        break

  return image_items

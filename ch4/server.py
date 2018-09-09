
# ---------------------------------------------------------------------------
# Flask server
# ---------------------------------------------------------------------------

import os, sys
from os.path import join
from glob import glob
import logging
from pathlib import Path
import json
import pickle
import uuid

from datetime import datetime
from flask import Flask, request, render_template, jsonify
import numpy as np
import imutils
from PIL import Image
import cv2 as cv

dir_megapixels = join(str(Path(os.path.abspath(__file__)).parent.parent))
sys.path.append(dir_megapixels)
from megapixels.utils import im_utils, file_utils
from megapixels.utils.logger_utils import LoggerUtils
from megapixels.settings import global_cfg as gcfg

from app.settings import app_cfg as cfg
from app.processors.proc_face import FaceRecognizer


# logging
if cfg.LOGFILE is not None:
  logging.basicConfig(level=cfg.LOGLEVEL, format=cfg.LOGFILE_FORMAT, 
    filename=cfg.LOGFILE)
else:
  logging.basicConfig(level=cfg.LOGLEVEL, format=cfg.LOGFILE_FORMAT)
log = logging.getLogger()

# load precomputed encodings
fr = FaceRecognizer()
fr.set_encodings(cfg.DEFAULT_ENCODINGS)

idx_mid = cfg.DEFAULT_POINTS//2


# create flask app
app = Flask(__name__,  static_url_path='', static_folder=cfg.DIR_STATIC_LOCAL,
          template_folder=cfg.DIR_TEMPLATES_LOCAL)


@app.route('/', defaults={'return_type': None}, methods=['GET', 'POST'])
@app.route('/<return_type>', methods=['GET', 'POST'])
def route_index(return_type=None):

  # NB: 
  # fp__flask refers to filepaths for flask
  # fp__local refers to filesystem paths

  if request.method == 'POST':
    # save upload
    file = request.files['img_query']
    img = Image.open(file.stream)  # PIL image

    # verify the image
    if img.size[0] < 200 or img.size[1] < 200:
      log.error('image too small')
      if matches is None:
        if return_type == 'json':
          try:
            return_error('aasd')
          except:
            pass
          return jsonify({'success': False, 'message': 'Image too small'})
        else:
          return render_template('index.html')

    # ensure it's RGB and not RGBA
    if img.format == 'PNG' and img.mode == 'RGBA':
      img = img.convert('RGB')

    # save it local filesystem
    now = file_utils.slugify(datetime.now().isoformat())
    name_uuid = uuid.uuid4()
    fname_uuid = '{}.jpg'.format(name_uuid)
    fp_upload_local = join(cfg.DIR_UPLOADS_LOCAL, fname_uuid)
    fp_upload_static = join(cfg.DIR_UPLOADS_STATIC, fname_uuid)
    img.save(fp_upload_local)

    # find face recognition matches
    matches = fr.match(fp_upload_local, num_matches=cfg.DEFAULT_MATCHES)
    if matches is None:
      if return_type == 'json':
        return jsonify({'success': False, 'message': 'No face detected'})
      else:
        return render_template('index.html')
  
    # create variables for Jinja templates
    match_items = []
    fp_match_static_all = []
    for match in matches:
      fp_match_static = join(cfg.DIR_DATASET_STATIC, match['person'], gcfg.DIRNAME_APPROVED, 
      '{}.jpg'.format(match['filename']))
      score = '{:.2f}'.format(100.0 * (match['score']))
      match_items.append({'filepath': fp_match_static, 
        'score': score, 'person': match['person']})
      fp_match_static_all.append(fp_match_static)

    match_best = matches[0]
    fp_match_local = join(cfg.DIR_DATASET_LOCAL, match_best['person'], gcfg.DIRNAME_APPROVED, 
      '{}.jpg'.format(match_best['filename']))


    # using best match, create hybrid face using openai/glow
    try:
      # check if var exists
      glow
    except:
      # load glow network (takes 5-10 seconds)
      from app.processors.glow import Glow
      glow = Glow()

    log.info('upload: {}'.format(fp_upload_local))
    log.info('match: {}'.format(fp_match_local))
    ims_glow = glow.generate(fp_upload_local, fp_match_local, cfg.DEFAULT_EXPAND, points=cfg.DEFAULT_POINTS)
    im_glow = ims_glow[idx_mid]

    glow_items = []
    fp_morph_static_all = []
    for idx, im in enumerate(ims_glow):
      idx_zfill= str(idx).zfill(2)
      fp_tmp_local = join(cfg.DIR_PROCESSED_LOCAL, 
        '{}_{}.jpg'.format(name_uuid, idx_zfill))
      fp_tmp_static = join(cfg.DIR_PROCESSED_STATIC, 
        '{}_{}.jpg'.format(name_uuid, idx_zfill))
      # write to local disk
      cv.imwrite(fp_tmp_local, im)
      glow_items.append({'filepath': fp_tmp_static, 'point': idx})
      fp_morph_static_all.append(fp_tmp_static)

    # prepare filepaths for Jinja tempaltes
    # NB: using cv in Flask can cause problems
    idx_zfill = str(idx_mid).zfill(2)
    fp_glow_local = join(cfg.DIR_PROCESSED_LOCAL, 
      '{}_{}.jpg'.format(name_uuid, idx_zfill))
    fp_glow_static = join(cfg.DIR_PROCESSED_STATIC, 
      '{}_{}.jpg'.format(name_uuid, idx_zfill))

    # return JSON if client wants, otherwise render HTML page
    if return_type == 'json':
      return jsonify({
        'success': True,
        'upload': fp_upload_static, 
        'morph': fp_glow_static,
        'match': fp_match_static,
        'morph_all': fp_morph_static_all,
        'match_all': fp_match_static_all})
    else:
      return render_template('index.html', 
        fp_upload=fp_upload_static,
        match_items=match_items,
        glow_items=glow_items)

  else:
    if return_type == 'json':
      return jsonify({'success': False})
    else:
      return render_template('index.html')


if __name__ == '__main__':
  app.run('0.0.0.0', port=5000)

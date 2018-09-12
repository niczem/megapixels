"""
Interpolate faces
"""

import os, sys
from os.path import join
from glob import glob
import logging
from pathlib import Path
from operator import itemgetter
from tqdm import tqdm
import shutil
import click
import json
import pickle
import itertools
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
from app.utils import dataset_utils
from app.processors.proc_face import HaarDetector, FaceRecognizer
from app.models.image_items import ImageFilterItem



# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
@click.group()
@click.option('--logfile', 'fp_log', type=str, default=None,
  help='Path to logfile or stdout')
@click.option('-v','--verbose', 'verbosity', count=True, default=3,
  show_default=True,
  help='Verbosity: -v DEBUG, -vv INFO, -vvv WARN, -vvvv ERROR, -vvvvv CRITICAL')
@click.pass_context
def cli(ctx, fp_log, verbosity):
  """Utility scripts"""
  ctx.obj['logger'] = LoggerUtils.configure(verbosity=verbosity, logfile=fp_log)


# ---------------------------------------------------------------------------
# Deduplicate directories of images
# ---------------------------------------------------------------------------
@cli.command('dedupe')
@click.option('-i', '--input', 'dir_in',
  default=cfg.DEFAULT_TARGETS,
  help='Director for input images')
@click.option('--confirm/--no-confirm', 'confirm', is_flag=True,
  help='Confirm face to ground truth image')
@click.option('-t', '--threshold', default=10,
  type=int,
  help='perceptual hash threshold')
@click.pass_context
def cmd_dedupe(ctx, dir_in, confirm, threshold):
  """Deduplicate images"""
  
  # init logger
  log = ctx.obj['logger']

  dirs_identity = glob(join(dir_in,'*'))
  dirs_identity.sort()
  dirs_identity = [x for x in dirs_identity if os.path.isdir(x)]
  
  # create list of files to process
  for dir_identity in dirs_identity[:4]:
    identity_name = Path(dir_identity).name
    log.info('---------------------------------------------------------')
    log.info('Processing: {}'.format(identity_name))
    im_list = file_utils.glob_images(dir_identity)
    image_items = [ImageFilterItem(x) for x in im_list]

    # dedupe using image hashes
    image_items = dataset_utils.filter_imagehash(image_items)  

    # reject images containing more than one face
    image_items = dataset_utils.filter_multiface(image_items)  
    
    # reject grayscal images
    image_items = dataset_utils.filter_grayscale(image_items)

    # TODO
    # filter items by pose to keep only frontal faces

    # if target available, remove non-target matches
    if confirm:
      log.info('filter identity')
      image_items = dataset_utils.filter_identity(image_items, 
        dir_identity, confirm=confirm)

    # ensure output directories
    dir_rejected = join(dir_identity, cfg.DIRNAME_REJECTED)
    file_utils.mkdirs(dir_rejected)
    dir_approved = join(dir_identity, cfg.DIRNAME_APPROVED)
    file_utils.mkdirs(dir_approved)

    # move/copy/write files to another directory
    for item in image_items:
      fp_src = item.fp_src
      if item.rejected:
        fp_dst = join(dir_rejected, Path(fp_src).name)
        shutil.copyfile(fp_src, fp_dst)
      else:
        # write the resized image
        im = im_utils.resize(item.im, 
          width=cfg.MAX_IMAGE_SIZE[0], height=cfg.MAX_IMAGE_SIZE[1])
        name_slugified = file_utils.slugify(Path(fp_src).stem)
        fp_dst = join(dir_approved, '{}.{}'.format(name_slugified, 'jpg'))
        cv.imwrite(fp_dst, im)

    # summarize
    log.info('Name: {}'.format(identity_name))
    log.info('Approved: {}'.format(len(dataset_utils.approved(image_items))))
    log.info('Rejected: {}'.format(len(dataset_utils.rejected(image_items))))

# ---------------------------------------------------------------------------
# Compute face recognition encodings
# ---------------------------------------------------------------------------
@cli.command('encode')
@click.option('-i', '--input', 'dir_in',
  default=cfg.DEFAULT_TARGETS,
  help='Directory where all identity directories are stored')
@click.option('-o', '--output', 'fp_out',
  default=cfg.DEFAULT_ENCODINGS,
  help="Filepath for output file (JSON or Pickle)")
@click.pass_context
def cmd_encode(ctx, dir_in, fp_out):
  """Generates face encodings for all identities"""
  log = ctx.obj['logger']

  dirs_identity = glob(join(dir_in, '*'))
  dirs_identity = [x for x in dirs_identity if os.path.isdir(x)]
  encodings = dataset_utils.generate_encodings(dirs_identity)
  log.info('total encodings: {:,}'.format(len(encodings)))

  ext = file_utils.get_ext(fp_out)
  if ext == 'json':
    with open(fp_out,'w') as fp:
      json.dump(encodings, fp, indent=2, sort_keys=True)
  elif ext == 'pkl':
    with open(fp_out, 'wb') as fp:
      pickle.dump(encodings, fp)
  else:
    log.error('"{}" is not a valid extension. Use json or pkl')


# ---------------------------------------------------------------------------
# Find match
# ---------------------------------------------------------------------------
# compute encoding of input, then match to dataset
@cli.command('match')
@click.option('-i', '--input', 'fp_in', type=str, required=True,
  help='Path to input ile to match')
@click.option('-e', '--encodings', 'fp_encodings', type=str, required=True,
  help='Path to input ile to match')
@click.option('-n', '--num-matches', default=3,
  help='Number of matches to show')
@click.option('--targets', 'dir_targets', type=str,required=True,
  help='Directory containing identities')
@click.option('-t', '--threshold', default=0.6, type=float,
  help='Facial recognition threshold')
@click.pass_context
def cmd_match(ctx, fp_in, fp_encodings, num_matches, dir_targets, threshold):
  """Finds best matches"""
  log = ctx.obj['logger']

  fr = FaceRecognizer()
  fr.set_encodings(fp_encodings)
  matches = fr.match(fp_in, num_matches=num_matches)

  for match in matches:
    log.info(match)

  # display as montage
  font = cv.FONT_HERSHEY_SIMPLEX
  bottomLeftCornerOfText = (10,500)
  fontScale = 0.8
  clr_pass = (0,255,0)
  clr_fail = (0,0,255)
  clr_bot = (0,0,0)
  lineType = 2


  ims = []
  for match in matches:
    fp_im = join(dir_targets, match['person'], gcfg.DIRNAME_APPROVED, '{}.jpg'.format(match['filename']))
    im = cv.imread(fp_im)
    im = im_utils.fit_image(im, (300,300))
    # force size with ImageFit
    txt = '{:.2f}'.format(100* float(match['score']) )
    im = cv.putText(im, txt, (21, 21), font, fontScale, clr_bot, lineType)
    clr = clr_fail if float(match['score']) > threshold else clr_pass
    im = cv.putText(im, txt, (20, 20), font, fontScale, clr, lineType)
    ims.append(im)

  im_query = cv.imread(fp_in)
  im_query = im_utils.fit_image(im_query, (300,300))
  cv.imshow('query', im_query)
  im_montage = im_utils.montage(ims, ncols=3)
  cv.imshow('results', im_montage)
  cv.waitKey(0)
  cv.destroyAllWindows()
    


# ---------------------------------------------------------------------------
# Glow best match
# ---------------------------------------------------------------------------
# compute encoding of input, then match to dataset, then generate glow
@cli.command('match_glow')
@click.option('-i', '--input', 'fp_query', type=str, required=True,
  help='Path to input ile to match')
@click.option('-e', '--encodings', 'fp_encodings', type=str, required=True,
  help='Path to input file to match')
@click.option('-o', '--output', 'dir_out', default=None, type=str,
  help='Path to output directory')
@click.option('--matches', 'num_matches', default=3,
  help='Number of matches to show')
@click.option('--targets', 'dir_targets', type=str,required=True,
  help='Directory containing identities')
@click.option('--threshold', default=0.6, type=float,
  help='Facial recognition threshold')
@click.option('--expand', default=0, type=int,
  help='Pixels to expand image crop (0-50')
@click.option('--points', default=3, type=int,
  help='Number of interpolation points (must be odd')
@click.option('--display', is_flag=True,
  help='Display result to screen')
@click.option('--montage', is_flag=True,
  help='Make interploation montage')
@click.option('--cols', 'ncols', default=3,
  help='Number of columns in montage image')
@click.pass_context
def cmd_match_glow(ctx, fp_query, fp_encodings, dir_out, num_matches,
 dir_targets, threshold, expand, points, display, montage, ncols):
  """Glow best matched face"""
  log = ctx.obj['logger']

  fr = FaceRecognizer()
  fr.set_encodings(fp_encodings)
  matches = fr.match(fp_query, num_matches=num_matches)
  if matches is None:
    log.error('input photo not useable. try a different photo')
    return

  match = matches[0]
  fp_match = join(dir_targets, match['person'], gcfg.DIRNAME_APPROVED, 
    '{}.jpg'.format(match['filename']))
  log.info('match: {}, score: {:.2f}'.format(match['person'], 
    100*float(match['score'])))
  log.info('setting up glow...')
  from app.processors.glow import Glow
  glow = Glow()
  ims_glow = glow.generate(fp_query, fp_match, expand, points=points)

  # get middle
  im_glow = ims_glow[len(ims_glow)//2]

  # create montage
  if montage:
    im_montage = im_utils.montage(ims_glow, ncols=ncols)

  # display to screen
  if display or not dir_out:
    cv.imshow('glow', im_glow)
    if montage:
      cv.imshow('montage', im_montage)
    cv.waitKey(0)
    cv.destroyAllWindows()

  # save result
  if dir_out:
    fpp_a = Path(fp_in_a)
    fpp_b = Path(fp_in_b)
    fp_glow = join(dir_out, '{}_x_{}.jpg'.format(fpp_a.stem, fpp_b.stem))
    fpp_glow = Path(fp_glow)
    cv.imwrite(fpp_glow, im_glow)
    if montage:
      fp_montage = join(dir_out, '{}_x_{}.jpg'.format(fpp_a.stem, fpp_b.stem))
      fpp_montage = Path(fp_montage)
      cv.imwrite(fpp_montage, im_glow)

  pass



# ---------------------------------------------------------------------------
# Glow: Image Synthesis
# ---------------------------------------------------------------------------
@cli.command('glow')
@click.option('-a', '--input_a','fp_in_a', required=True, type=str,
  help='Path to input file A')
@click.option('-b', '--input_b','fp_in_b', required=True, type=str,
  help='Path to input file B')
@click.option('-o', '--output','dir_out', default=None, type=str,
  help='Path to output directory')
@click.option('-e', '--expand', default=0, type=int,
  help='Pixels to expand image crop (0-50')
@click.option('-p', '--points', default=3, type=int,
  help='Number of interpolation points (must be odd')
@click.option('-d', '--display', is_flag=True,
  help='Display result to screen')
@click.option('-m', '--montage', is_flag=True,
  help='Make interploation montage')
@click.option('--cols', 'ncols', default=3,
  help='Number of columns in montage image')
@click.pass_context
def cmd_glow_simple(ctx, fp_in_a, fp_in_b, dir_out, 
    expand, points, display, montage, ncols):
  """Synthesize new face using openai/glow"""
  log = ctx.obj['logger']

  # generate glow face
  from app.processors.glow import Glow
  glow = Glow()
  ims_glow = glow.generate(fp_in_a, fp_in_b, expand, points=points)

  # get middle
  im_glow = ims_glow[len(ims_glow)//2]

  # create montage
  if montage:
    im_montage = im_utils.montage(ims_glow, ncols=ncols)

  # display to screen
  if display or not dir_out:
    cv.imshow('glow', im_glow)
    if montage:
      cv.imshow('montage', im_montage)
    cv.waitKey(0)
    cv.destroyAllWindows()

  # save result
  if dir_out:
    fpp_a = Path(fp_in_a)
    fpp_b = Path(fp_in_b)
    fp_glow = join(dir_out, '{}_x_{}.jpg'.format(fpp_a.stem, fpp_b.stem))
    fpp_glow = Path(fp_glow)
    cv.imwrite(fpp_glow, im_glow)
    if montage:
      fp_montage = join(dir_out, '{}_x_{}.jpg'.format(fpp_a.stem, fpp_b.stem))
      fpp_montage = Path(fp_montage)
      cv.imwrite(fpp_montage, im_glow)




# ---------------------------------------------------------------------------
# Switching Eds: Face swapping
# ---------------------------------------------------------------------------
@cli.command('swap')
@click.option('-a', '--input_a','fp_in_a', required=True, type=str,
  help='Path to input file A')
@click.option('-b', '--input_b','fp_in_b', required=True, type=str,
  help='Path to input file B')
@click.option('-o', '--output','dir_out', default=None, type=str,
  help='Path to output directory')
@click.option('-d', '--display', is_flag=True,
  help='Display result to screen')
@click.pass_context
def cmd_faceswap_simple(ctx, fp_in_a, fp_in_b, dir_out, display):
  """Clone person A's face to person B"""

  log = ctx.obj['logger']
  
  from app.processors.faceswap import Faceswap
  faceswap = Faceswap()
 
  log.info('{} --> {}'.format(fp_in_a, fp_in_b))
  im_synth = faceswap.generate(fp_in_a, fp_in_b)
  
  # save or display output
  if dir_out:
    # get name
    fp_im_c = join(dir_out, '{}_x_{}.jpg'.format(Path(a).stem, Path(b).stem))
    cv.imwrite(fp_im_c, im_synth)
  if display:
    # display to screen
    im_synth_rgb = im_utils.bgr2rgb(im_synth)
    im_pil = im_utils.ensure_pil(im_synth_rgb)
    im_pil.show()



# ---------------------------------------------------------------------------
# Change directory names to a slugified name
# ---------------------------------------------------------------------------
@cli.command('slugify')
@click.option('-i', '--input', 'dir_in', required=True,
  help='Directory where all identity directories are stored')
@click.option('--dry-run', is_flag=True,
  help="Dry run with print statements")
@click.pass_context
def cmd_slugify(ctx, dir_in, dry_run):
  """Slugifies identity directories"""
  log = ctx.obj['logger']
  dirs = glob(join(dir_in,'*'))
  dirs = [x for x in dirs if os.path.isdir(x)]

  for dir_src in dirs:
    dirp_src = Path(dir_src)
    dir_src_name = dirp_src.name
    slug = file_utils.slugify(dir_src_name)
    dir_dst = join(str(dirp_src.parent), slug)
    if dry_run:
      log.info('(Dry Run) {} --> {}'.format(dir_src_name, Path(dir_dst).name))
    else:
      log.info('Moved {} --> {}'.format(dir_src_name, Path(dir_dst).name))
      shutil.move(dir_src, dir_dst)



# ---------------------------------------------------------------------------
# Flask server
# ---------------------------------------------------------------------------
@cli.command('server')
@click.option('--port', default=5000,
  help='Flask server port')
@click.option('--address', default='0.0.0.0',
  help='Flask server IP address')
@click.option('-e', '--encodings', 'fp_encodings', type=str, required=True,
  default=cfg.DEFAULT_ENCODINGS,
  help='Path to input file to match')
@click.option('-n', '--num-matches', default=cfg.DEFAULT_MATCHES,
  help='Number of matches to show')
@click.option('--expand', default=cfg.DEFAULT_EXPAND, type=int,
  help='Pixels to expand image crop (0-50')
@click.option('--points', default=str(cfg.DEFAULT_POINTS), type=click.Choice(['3','5','7','9']),
  help='Number of interpolation points, only for montages')
@click.pass_context
def cmd_server(ctx, port, address, fp_encodings, num_matches, 
  expand, points):
  """Runs server"""
  log = ctx.obj['logger']
  log.info('run server')

  # load precomputed encodings
  fr = FaceRecognizer()
  fr.set_encodings(fp_encodings)

  idx_mid = int(points)//2

  # create flask app
  app = Flask(__name__,  static_url_path='', static_folder=cfg.DIR_STATIC_LOCAL,
            template_folder=cfg.DIR_TEMPLATES_LOCAL)

  def return_error(msg):
    log.error('from outside')

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

      if img.format == 'PNG' and img.mode == 'RGBA':
        img = img.convert('RGB')

      now = file_utils.slugify(datetime.now().isoformat())
      name_uuid = uuid.uuid4()
      fname_uuid = '{}.jpg'.format(name_uuid)
      fp_upload_local = join(cfg.DIR_UPLOADS_LOCAL, fname_uuid)
      fp_upload_static = join(cfg.DIR_UPLOADS_STATIC, fname_uuid)
      img.save(fp_upload_local)

      # find matches
      matches = fr.match(fp_upload_local, num_matches=num_matches)
      if matches is None:
        if return_type == 'json':
          return jsonify({'success': False, 'message': 'No face detected'})
        else:
          return render_template('index.html')
    
      match_items = []
      fp_match_static_all = []
      for match in matches:
        fp_match_static = join(cfg.DIR_DATASET_STATIC, match['person'], cfg.DIRNAME_APPROVED, 
        '{}.jpg'.format(match['filename']))
        score = '{:.2f}'.format(100.0 * (match['score']))
        match_items.append({'filepath': fp_match_static, 
          'score': score, 'person': match['person']})
        fp_match_static_all.append(fp_match_static)

      match_best = matches[0]
      fp_match_local = join(cfg.DIR_DATASET_LOCAL, match_best['person'], cfg.DIRNAME_APPROVED, 
        '{}.jpg'.format(match_best['filename']))

      # glow morph
      try:
        # check if var exists
        glow
      except:
        # load glow network (takes 5-10 seconds)
        from app.processors.glow import Glow
        glow = Glow()

      ims_glow = glow.generate(fp_upload_local, fp_match_local, expand, points=int(points))
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

      # NB: using cv in Flask can cause problems
      idx_zfill = str(idx_mid).zfill(2)
      fp_glow_local = join(cfg.DIR_PROCESSED_LOCAL, 
        '{}_{}.jpg'.format(name_uuid, idx_zfill))
      fp_glow_static = join(cfg.DIR_PROCESSED_STATIC, 
        '{}_{}.jpg'.format(name_uuid, idx_zfill))

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


  app.run(address, port=port)



# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
@cli.command('test')
@click.pass_context
def cmd_test(ctx):
  """Test code snippets here"""
  log = ctx.obj['logger']
  
  log.info('testing...')
  for k, v in ctx.obj.items():
    print('{}: {}'.format(k, v))

  from app.processors.glow import Glow
  glow = Glow()
  log.info('if you can see this message, it probably works')
  


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == '__main__':
  cli(obj={})













# ---------------------------------------------------------------------------
# Faceswap best match
# ---------------------------------------------------------------------------
# not implemented

# ---------------------------------------------------------------------------
# Compare recognition scores
# ---------------------------------------------------------------------------
# not implemented

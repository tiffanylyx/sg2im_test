#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sg2im.data import imagenet_deprocess_batch
#from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard
from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager
import sys
torch.backends.cudnn.benchmark = True

VG_DIR = os.path.expanduser('datasets/vg')
COCO_DIR = os.path.expanduser('datasets/coco')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# Generator losses
parser.add_argument('--mask_loss_weight', default=0, type=float)
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--predicate_pred_loss_weight', default=0, type=float) # DEPRECATED

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)


def check_args(args):
  H, W = args.image_size
  for _ in args.refinement_network_dims[1:]:
    H = H // 2
  if H == 0:
    raise ValueError("Too many layers in refinement network")


def build_model(args, vocab):
  if args.checkpoint_start_from is not None:
    checkpoint = torch.load(args.checkpoint_start_from)
    kwargs = checkpoint['model_kwargs']
    model = Sg2ImModel(**kwargs)
    raw_state_dict = checkpoint['model_state']
    state_dict = {}
    for k, v in raw_state_dict.items():
      if k.startswith('module.'):
        k = k[7:]
      state_dict[k] = v
    model.load_state_dict(state_dict)
  else:
    kwargs = {
      'vocab': vocab,
      'image_size': args.image_size,
      'embedding_dim': args.embedding_dim,
      'gconv_dim': args.gconv_dim,
      'gconv_hidden_dim': args.gconv_hidden_dim,
      'gconv_num_layers': args.gconv_num_layers,
      'mlp_normalization': args.mlp_normalization,
      'refinement_dims': args.refinement_network_dims,
      'normalization': args.normalization,
      'activation': args.activation,
      'mask_size': args.mask_size,
      'layout_noise_dim': args.layout_noise_dim,
    }
    model = Sg2ImModel(**kwargs)
  return model, kwargs

def build_coco_dsets(args):
  dset_kwargs = {
    'image_dir': args.coco_train_image_dir,
    'instances_json': args.coco_train_instances_json,
    'stuff_json': args.coco_train_stuff_json,
    'stuff_only': args.coco_stuff_only,
    'image_size': args.image_size,
    'mask_size': args.mask_size,
    'max_samples': args.num_train_samples,
    'min_object_size': args.min_object_size,
    'min_objects_per_image': args.min_objects_per_image,
    'instance_whitelist': args.instance_whitelist,
    'stuff_whitelist': args.stuff_whitelist,
    'include_other': args.coco_include_other,
    'include_relationships': args.include_relationships,
  }
  train_dset = CocoSceneGraphDataset(**dset_kwargs)
  num_objs = train_dset.total_objects()
  num_imgs = len(train_dset)
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  dset_kwargs['image_dir'] = args.coco_val_image_dir
  dset_kwargs['instances_json'] = args.coco_val_instances_json
  dset_kwargs['stuff_json'] = args.coco_val_stuff_json
  dset_kwargs['max_samples'] = args.num_val_samples
  val_dset = CocoSceneGraphDataset(**dset_kwargs)

  assert train_dset.vocab == val_dset.vocab
  vocab = json.loads(json.dumps(train_dset.vocab))

  return vocab, train_dset, val_dset


def build_vg_dsets(args):
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = VgSceneGraphDataset(**dset_kwargs)
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = VgSceneGraphDataset(**dset_kwargs)
  
  return vocab, train_dset, val_dset


def build_loaders(args):
  if args.dataset == 'vg':
    vocab, train_dset, val_dset = build_vg_dsets(args)
    collate_fn = vg_collate_fn
  elif args.dataset == 'coco':
    vocab, train_dset, val_dset = build_coco_dsets(args)
    collate_fn = coco_collate_fn

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': True,
    'collate_fn': collate_fn,
  }
  train_loader = DataLoader(train_dset, **loader_kwargs)
  
  loader_kwargs['shuffle'] = args.shuffle_val
  val_loader = DataLoader(val_dset, **loader_kwargs)
  return vocab, train_loader, val_loader


def check_model(args, t, loader, model):
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor
  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 7:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1] 

      # Run the model as it has been run during training
      model_masks = masks
      boxes_pred = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)

      skip_pixel_loss = False
      loss = calculate_model_losses(boxes, boxes_pred)

      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break
    '''
    samples = {}
    samples['gt_img'] = imgs

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
    samples['gt_box'] = model_out

    model_out = model(objs, triples, obj_to_img)
    samples['pred_box'] = model_out[0]

    for k, v in samples.items():
      samples[k] = imagenet_deprocess_batch(v)
    '''
    mean_losses = loss
    avg_iou = total_iou / total_boxes

  batch_data = {
    'objs': objs.detach().cpu().clone(),
    'boxes_gt': boxes.detach().cpu().clone(),
    'triples': triples.detach().cpu().clone(),
    'obj_to_img': obj_to_img.detach().cpu().clone(),
    'triple_to_img': triple_to_img.detach().cpu().clone(),
    'boxes_pred': boxes_pred.detach().cpu().clone(),
  }
  out = [mean_losses, batch_data, avg_iou]

  return tuple(out)


def calculate_model_losses(bbox, bbox_pred):
  loss_bbox = F.mse_loss(bbox_pred, bbox)
  return loss_bbox


def main(args):
  print(args)
  check_args(args)
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor

  vocab, train_loader, val_loader = build_loaders(args)
  model, model_kwargs = build_model(args, vocab)
  model.type(float_dtype)
  print(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  restore_path = None
  if args.restore_from_checkpoint:
    restore_path = '%s_with_model.pt' % args.checkpoint_name
    restore_path = os.path.join(args.output_dir, restore_path)
  if restore_path is not None and os.path.isfile(restore_path):
    print('Restoring from checkpoint:')
    print(restore_path)
    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])

    t = checkpoint['counters']['t']
    if 0 <= args.eval_mode_after <= t:
      model.eval()
    else:
      model.train()
    epoch = checkpoint['counters']['epoch']
  else:
    t, epoch = 0, 0
    checkpoint = {
      'args': args.__dict__,
      'vocab': vocab,
      'model_kwargs': model_kwargs,
      'losses_ts': [],
      'losses': [],
      'd_losses': [],
      'checkpoint_ts': [],
      'train_batch_data': [], 
      'train_samples': [],
      'train_iou': [],
      'val_batch_data': [], 
      'val_samples': [],
      'val_losses': [],
      'val_iou': [], 
      'norm_d': [], 
      'norm_g': [],
      'counters': {
        't': None,
        'epoch': None,
      },
      'model_state': None, 'model_best_state': None, 'optim_state': None,
      'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
      'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
      'best_t': [],
    }

  while True:
    if t >= args.num_iterations:
      break
    epoch += 1
    print('Starting epoch %d' % epoch)
    
    for batch in train_loader:
      if t == args.eval_mode_after:
        print('switching to eval mode')
        model.eval()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
      t += 1
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      else:
        assert False
      predicates = triples[:, 1]

      with timeit('forward', args.timing):
        model_boxes = boxes
        boxes_pred = model(objs, triples, obj_to_img,
                          boxes_gt=model_boxes, masks_gt=None)
      with timeit('loss', args.timing):
        # Skip the pixel loss if using GT boxes
        skip_pixel_loss = (model_boxes is None)
        loss = calculate_model_losses(boxes, boxes_pred)

      if not math.isfinite(loss):
        print('WARNING: Got loss = NaN, not backpropping')
        continue

      optimizer.zero_grad()
      loss.requires_grad=True
      with timeit('backward', args.timing):
        loss.backward()
      optimizer.step()
      total_loss_d = None
      ac_loss_real = None
      ac_loss_fake = None
      d_losses = []

      if t % args.print_every == 0:
        print('t = %d / %d' % (t, args.num_iterations))
        checkpoint['losses'].append(loss)
        checkpoint['losses_ts'].append(t)

      if t % args.checkpoint_every == 0:
        print('checking on train')
        train_results = check_model(args, t, train_loader, model)
        t_losses, t_batch_data, t_avg_iou = train_results

        checkpoint['train_batch_data'].append(t_batch_data)
        #checkpoint['train_samples'].append(t_samples)
        checkpoint['checkpoint_ts'].append(t)
        checkpoint['train_iou'].append(t_avg_iou)

        print('checking on val')
        val_results = check_model(args, t, val_loader, model)
        val_losses, val_batch_data, val_avg_iou = val_results
        print('val_losses',val_losses)
        #checkpoint['val_samples'].append(val_samples)
        checkpoint['val_batch_data'].append(val_batch_data)
        checkpoint['val_iou'].append(val_avg_iou)

        print('train iou: ', t_avg_iou)
        print('val iou: ', val_avg_iou)

        for k, v in val_losses.items():
          checkpoint['val_losses'][k].append(v)
        checkpoint['model_state'] = model.state_dict()

        checkpoint['optim_state'] = optimizer.state_dict()
        checkpoint['counters']['t'] = t
        checkpoint['counters']['epoch'] = epoch
        checkpoint_path = os.path.join(args.output_dir,
                              '%s_with_model.pt' % args.checkpoint_name)
        print('Saving checkpoint to ', checkpoint_path)
        torch.save(checkpoint, checkpoint_path)

        # Save another checkpoint without any model or optim state
        checkpoint_path = os.path.join(args.output_dir,
                              '%s_no_model.pt' % args.checkpoint_name)
        key_blacklist = ['model_state', 'optim_state', 'model_best_state',
                         'd_obj_state', 'd_obj_optim_state', 'd_obj_best_state',
                         'd_img_state', 'd_img_optim_state', 'd_img_best_state']
        small_checkpoint = {}
        for k, v in checkpoint.items():
          if k not in key_blacklist:
            small_checkpoint[k] = v
        torch.save(small_checkpoint, checkpoint_path)


if __name__ == '__main__':
  sys.stdout.flush()
  args = parser.parse_args()
  main(args)


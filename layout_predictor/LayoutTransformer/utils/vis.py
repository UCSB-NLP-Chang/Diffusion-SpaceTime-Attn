import tempfile, os
import cv2
import torch
import numpy as np
import io
import matplotlib as mpl
mpl.use('Agg')
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from imageio import imread


def draw_scene_graph(objs, triples, vocab=None, **kwargs):
  """
  Use GraphViz to draw a scene graph. If vocab is not passed then we assume
  that objs and triples are python lists containing strings for object and
  relationship names.
  Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
  sudo apt-get install graphviz
  """
  output_filename = kwargs.pop('output_filename', 'graph.png')
  orientation = kwargs.pop('orientation', 'V')
  edge_width = kwargs.pop('edge_width', 6)
  arrow_size = kwargs.pop('arrow_size', 1.5)
  binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
  ignore_dummies = kwargs.pop('ignore_dummies', True)
  
  if orientation not in ['V', 'H']:
    raise ValueError('Invalid orientation "%s"' % orientation)
  rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

  if vocab is not None:
    # Decode object and relationship names
    assert torch.is_tensor(objs)
    assert torch.is_tensor(triples)
    objs_list, triples_list = [], []
    for i in range(objs.size(0)):
      objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
    for i in range(triples.size(0)):
      s = triples[i, 0].item()
      p = vocab['pred_name_to_idx'][triples[i, 1].item()]
      o = triples[i, 2].item()
      triples_list.append([s, p, o])
    objs, triples = objs_list, triples_list

  # General setup, and style for object nodes
  lines = [
    'digraph{',
    'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
    'rankdir=%s' % rankdir,
    'nodesep="0.5"',
    'ranksep="0.5"',
    'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
    'node [fillcolor="lightpink1"]',
  ]
  # Output nodes for objects
  for i, obj in enumerate(objs):
    if ignore_dummies and obj == '__image__':
      continue
    lines.append('%d [label="%s"]' % (i, obj))

  # Output relationships
  next_node_id = len(objs)
  lines.append('node [fillcolor="lightblue1"]')
  for s, p, o in triples:
    if ignore_dummies and p == '__in_image__':
      continue
    lines += [
      '%d [label="%s"]' % (next_node_id, p),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
        s, next_node_id, edge_width, arrow_size, binary_edge_weight),
      '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
        next_node_id, o, edge_width, arrow_size, binary_edge_weight)
    ]
    next_node_id += 1
  lines.append('}')

  # Now it gets slightly hacky. Write the graphviz spec to a temporary
  # text file
  ff, dot_filename = tempfile.mkstemp()
  with open(dot_filename, 'w') as f:
    for line in lines:
      f.write('%s\n' % line)
  os.close(ff)

  # Shell out to invoke graphviz; this will save the resulting image to disk,
  # so we read it, delete it, then return it.
  output_format = os.path.splitext(output_filename)[1][1:]
#   print('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
  os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
  os.remove(dot_filename)
  img = imread(output_filename)
  os.remove(output_filename)

  return img



if __name__ == '__main__':
  o_idx_to_name = ['cat', 'dog', 'hat', 'skateboard']
  p_idx_to_name = ['riding', 'wearing', 'on', 'next to', 'above']
  o_name_to_idx = {s: i for i, s in enumerate(o_idx_to_name)}
  p_name_to_idx = {s: i for i, s in enumerate(p_idx_to_name)}
  vocab = {
    'object_idx_to_name': o_idx_to_name,
    'object_name_to_idx': o_name_to_idx,
    'pred_idx_to_name': p_idx_to_name,
    'pred_name_to_idx': p_name_to_idx,
  }

  objs = [
    'cat',
    'cat',
    'skateboard',
    'hat',
  ]
  objs = torch.LongTensor([o_name_to_idx[o] for o in objs])
  triples = [
    [0, 'next to', 1],
    [0, 'riding', 2],
    [1, 'wearing', 3],
    [3, 'above', 2],
  ]
  triples = [[s, p_name_to_idx[p], o] for s, p, o in triples]
  triples = torch.LongTensor(triples)

  draw_scene_graph(objs, triples, vocab, orientation='V')
#!/usr/bin/env python

import rospy
import random
import math
import numpy
from timeit import default_timer as timer
from utilities import RvizHandler
from utilities import Print
from utilities import Cffi
from sets import Set
import scipy.misc
import time

from skimage.morphology import skeletonize

#scipy.misc.imsave('/home/manos/Desktop/test.png', self.coverage)

class Topology:

  # Constructor
  def __init__(self):
    self.numpy_implementation = rospy.get_param("numpy_implementation")
    Print.art_print("Fast numpy implementation =  %r" % self.numpy_implementation, Print.GREEN)

  def skeletonizationCffi(self, ogm, origin, resolution, ogml):

    if self.numpy_implementation == True:
      local = numpy.zeros(ogm.shape)

      local[ogm < 49] = 1
      skeleton = Cffi.thinning(local, ogml)
      skeleton = Cffi.prune(skeleton, ogml, 10)
      viz = (numpy.array(numpy.where(skeleton == 1)).T * resolution + [origin['x'], origin['y']]).tolist()

    else:
      width = ogm.shape[0]
      height = ogm.shape[1]

      local = numpy.zeros(ogm.shape)

      for i in range(0, width):
        for j in range(0, height):
          if ogm[i][j] < 49:
            local[i][j] = 1

      skeleton = Cffi.thinning(local, ogml)
      skeleton = Cffi.prune(skeleton, ogml, 10)

      viz = []
      for i in range(0, width):
        for j in range(0, height):
          if skeleton[i][j] == 1:
            viz.append([i * resolution + origin['x'], j * resolution + origin['y']])

    RvizHandler.printMarker(\
              viz,\
              1, # Type: Arrow
              0, # Action: Add
              "map", # Frame
              "art_skeletonization_cffi", # Namespace
              [0.5, 0, 0, 0.5], # Color RGBA
              0.05 # Scale
          )

    return skeleton

  def skeletonization(self, ogm, origin, resolution, ogml):

    if self.numpy_implementation == True:
      useful_ogm = ogm[ogml['min_x']:ogml['max_x'], ogml['min_y']:ogml['max_y']]

      useful_local = numpy.zeros(useful_ogm.shape)

      useful_local[useful_ogm < 49] = 1

      skeleton = skeletonize(useful_local)
      skeleton = self.pruning(skeleton, 10)

      # padding
      skeleton_final = numpy.zeros(ogm.shape)
      skeleton_final[ogml['min_x']:ogml['max_x'], ogml['min_y']:ogml['max_y']] = skeleton

      viz = (numpy.array(numpy.where(skeleton_final == 1)).T * resolution + [origin['x'], origin['y']]).tolist()
    else:
      width = ogm.shape[0]
      height = ogm.shape[1]

      useful_ogm = ogm[ogml['min_x']:ogml['max_x'], ogml['min_y']:ogml['max_y']]
      useful_width = useful_ogm.shape[0]
      useful_height = useful_ogm.shape[1]

      local = numpy.zeros(ogm.shape)
      useful_local = numpy.zeros(useful_ogm.shape)

      for i in range(0, useful_width):
        for j in range(0, useful_height):
          if useful_ogm[i][j] < 49:
            useful_local[i][j] = 1

      skeleton = skeletonize(useful_local)
      skeleton = self.pruning(skeleton, 10)

      # padding
      skeleton_final = numpy.zeros(ogm.shape)
      skeleton_final[ogml['min_x']:ogml['max_x'], ogml['min_y']:ogml['max_y']] = skeleton

      viz = []
      for i in range(0, width):
        for j in range(0, height):
          if skeleton_final[i][j] == 1:
            viz.append([i * resolution + origin['x'], j * resolution + origin['y']])

    RvizHandler.printMarker(\
            viz,\
            1, # Type: Arrow
            0, # Action: Add
            "map", # Frame
            "art_skeletonization", # Namespace
            [0.5, 0, 0, 0.5], # Color RGBA
            0.05 # Scale
        )


    #scipy.misc.imsave('/home/manos/Desktop/test.png', skeleton_final)
    return skeleton_final

  def topologicalNodes(self, ogm, skeleton, coverage, origin, resolution, brush, ogm_limits):
    nodes = []
    
    width = ogm.shape[0]
    height = ogm.shape[1]

    if self.numpy_implementation == True:
      index = numpy.where((ogm[1:width - 1, 1: height - 1] <= 49) & \
                          (brush[1:width - 1, 1: height - 1] > 3) & \
                          (skeleton[1:width - 1, 1: height - 1] == 1) & \
                          (coverage[1:width - 1, 1: height - 1] != 100))

      for i in range(len(index[0])):
        c = numpy.sum(skeleton[index[0][i]:index[0][i] + 3, index[1][i]:index[1][i] + 3])
        if (c == 2 or c == 4):
          nodes.append([index[0][i] + 1, index[1][i] + 1])

      for i in range(len(nodes) - 2, -1, -1):
        n1 = nodes[i]

        j = i + 1
        while j < len(nodes):
          n2 = nodes[j]
          if math.pow(n1[0] - n2[0], 2) + math.pow(n1[1] - n2[1], 2) < 25:
            del nodes[i]
          else:
            j += 1
    else:
      for i in range(1, width - 1):
        for j in range(1, height - 1):
          if ogm[i][j] <= 49 and brush[i][j] > 3 and skeleton[i][j] == 1 and coverage[i][j] != 100:
            c = 0
            for ii in range(-1, 2):
              for jj in range(-1, 2):
                c = c + skeleton[i + ii][j + jj]

            if (c == 2 or c == 4): # and coverage etc
              nodes.append([i, j])

      # minimize number of nodes by erasing
      change = True
      while change:
        change = False
        for i in range(0, len(nodes)):
          for j in range(0, len(nodes)):
            if i == j:
              continue
            n1 = nodes[i]
            n2 = nodes[j]
            if math.pow(n1[0] - n2[0], 2) + math.pow(n1[1] - n2[1], 2) < 25:
              change = True
              del nodes[i]
              break
          if change:
            break

    return nodes

  def pruning(self, img, n):
    for k in range(0, n):
      tmp_img = numpy.copy(img)
      for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
          if img[i][j] == 1:
            c = 0
            for ii in range(-1, 2):
              for jj in range(-1, 2):
                c = c + img[i + ii][j + jj]
            if c == 2:
              tmp_img[i][j] = 0
      img = numpy.copy(tmp_img)
    return img


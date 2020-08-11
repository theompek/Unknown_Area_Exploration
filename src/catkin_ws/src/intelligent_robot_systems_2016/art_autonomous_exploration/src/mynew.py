
#!/usr/bin/env python

import rospy
import random
import math
import time
import numpy as np
from timeit import default_timer as timer
from utilities import RvizHandler
from utilities import OgmOperations
from utilities import Print
from brushfires import Brushfires
from topology import Topology
import scipy
from path_planning import PathPlanning


# Class for selecting the next best target
class TargetSelection:

    # Constructor
    def __init__(self, selection_method):
        self.goals_position = []
        self.goals_value = []
        self.omega = 0.0
        self.radius = 0
        self.method = selection_method
        self.previous_target = [-1, -1]

        self.brush = Brushfires()
        self.topo = Topology()
        self.path_planning = PathPlanning()

    def selectTarget(self, init_ogm, coverage, robot_pose, origin,resolution, force_random=False):

        target = [-1, -1]

        ######################### NOTE: QUESTION  ##############################
        # Implement a smart way to select the next target. You have the
        # following tools: ogm_limits, Brushfire field, OGM skeleton,
        # topological nodes.

        # Find only the useful boundaries of OGM. Only there calculations
        # have meaning
        ogm_limits = OgmOperations.findUsefulBoundaries(init_ogm, origin, resolution)

        # Blur the OGM to erase discontinuities due to laser rays
        ogm = OgmOperations.blurUnoccupiedOgm(init_ogm, ogm_limits)

        # Calculate Brushfire field
        tinit = time.time()
        brush = self.brush.obstaclesBrushfireCffi(ogm, ogm_limits)
        Print.art_print("Brush time: " + str(time.time() - tinit), Print.ORANGE)

        # Calculate skeletonization
        tinit = time.time()
        skeleton = self.topo.skeletonizationCffi(ogm, \
                                                 origin, resolution, ogm_limits)
        Print.art_print("Skeletonization time: " + str(time.time() - tinit), Print.ORANGE)

        # Find topological graph
        tinit = time.time()
        nodes = self.topo.topologicalNodes(ogm, skeleton, coverage, origin, \
                                           resolution, brush, ogm_limits)
        Print.art_print("Topo nodes time: " + str(time.time() - tinit), Print.ORANGE)

        # Visualization of topological nodes
        vis_nodes = []
        for n in nodes:
            vis_nodes.append([
                n[0] * resolution + origin['x'],
                n[1] * resolution + origin['y']
            ])
        RvizHandler.printMarker( \
            vis_nodes, \
            1,  # Type: Arrow
            0,  # Action: Add
            "map",  # Frame
            "art_topological_nodes",  # Namespace
            [0.3, 0.4, 0.7, 0.5],  # Color RGBA
            0.1  # Scale
        )

        # Random point
        if self.method == 'random' or force_random == True:
            target = self.selectRandomTarget(ogm, coverage, brush, ogm_limits)

        # Cost_based target selection
        if self.method == 'cost_based_target_selection' and force_random == False:
            nextTarget = self.costBasedTargetSelection(coverage, brush, robot_pose, resolution, origin, nodes)

            if nextTarget is not None:
                # Check if the next target is the same as the previous
                distance = math.hypot(nextTarget[0] - self.previous_target[0], nextTarget[1] - self.previous_target[1])
                if distance > 6:
                    target = nextTarget
                else:
                    target = self.selectRandomTarget(ogm, coverage, brush, ogm_limits)
            else:
                # No target found. Choose a random
                target = self.selectRandomTarget(ogm, coverage, brush, ogm_limits)

        self.previous_target = target
        return target

    def costBasedTargetSelection(self, coverage, brush, robot_pose, resolution, origin, nodes):
        tinit = time.time()
        # Get the robot pose in pixels
        [rx, ry] = [ int(round(robot_pose['x_px'] - origin['x'] / resolution)), int(round(robot_pose['y_px'] - origin['y'] / resolution))]

        target_cost = []

        for current_node, node in enumerate(nodes):
            # Estimate the subgoals
            sub_goals = self.myflip(self.path_planning.createPath([rx, ry], node, resolution), 0)

            # Check if it found a path
            if sub_goals.shape[0] > 2:
                # Vectors of the path
                vectors = sub_goals[1:, :] - sub_goals[:-1, :]

                # Calculate paths weighted distance
                # vectorsMean = vectors.mean(axis=0)
                # vectorsVar = vectors.var(axis=0)
                # weightCoeff = 1 / (1 - np.exp(-np.sum((vectors - vectorsMean)**2 / (2 * (vectorsVar + 1e-3)), axis=1)) + 1e-3)
                # weightDists = np.sum(weightCoeff + dists)
                all_distances = np.sqrt(np.einsum('ij,ij->i', vectors, vectors))
                distances_cost = np.sum(all_distances)

                # Topological weight
                topological_cost = brush[node[0], node[1]]

                # Cosine of the angles
                angles_cosine = np.sum(vectors[1:, :] * vectors[:-1, :], axis=1) / np.linalg.norm(vectors[1:, :], axis=1) / np.linalg.norm(vectors[:-1, :], axis=1)

                # Sum of all angles
                turn_cost = np.sum(abs(np.arccos(np.clip(angles_cosine, -1, 1))))

                # Calculate the coverage weight
                pathIndex = np.rint(sub_goals).astype(int)
                coverage_cost = 1 - np.sum(coverage[pathIndex[:, 0], pathIndex[:, 1]]) / (sub_goals.shape[0] * 100)

                target_cost.append([current_node, distances_cost, topological_cost, turn_cost, coverage_cost])

        if len(target_cost) > 0:
            weight = np.array(target_cost)

            # Normalize the weights
            weight[:, 1:] = 1 - ((weight[:, 1:] - np.min(weight[:, 1:], axis=0)) / \
                                 (np.max(weight[:, 1:], axis=0) - np.min(weight[:, 1:], axis=0)))

            # Calculatete the final weights
            finalWeights = 8 * weight[:, 2] + 4 * weight[:, 1] + 2 * weight[:, 4] + weight[:, 3]

            # Find the best path
            index = int(weight[max(xrange(len(finalWeights)), key=finalWeights.__getitem__)][0])

            target = nodes[index]

            Print.art_print("Smart target selection time: " + str(time.time() - tinit), \
                            Print.ORANGE)

            return target
        else:
            Print.art_print("Smart target selection failed!!! Time: " + str(time.time() - tinit), \
                            Print.ORANGE)

            return None

    ########################################################################

    def selectRandomTarget(self, ogm, coverage, brushogm, ogm_limits):
        # The next target in pixels
        tinit = time.time()
        next_target = [0, 0]
        found = False
        while not found:
            x_rand = random.randint(0, ogm.shape[0] - 1)
            y_rand = random.randint(0, ogm.shape[1] - 1)
            if ogm[x_rand][y_rand] < 50 and coverage[x_rand][y_rand] < 50 and \
                    brushogm[x_rand][y_rand] > 5:
                next_target = [x_rand, y_rand]
                found = True
        Print.art_print("Select random target time: " + str(time.time() - tinit), \
                        Print.ORANGE)
        return next_target

    def myflip(self, m, axis):
        if not hasattr(m, 'ndim'):
            m = np.asarray(m)
        indexer = [slice(None)] * m.ndim
        try:
            indexer[axis] = slice(None, None, -1)
        except IndexError:
            raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                             % (axis, m.ndim))
        return m[tuple(indexer)]
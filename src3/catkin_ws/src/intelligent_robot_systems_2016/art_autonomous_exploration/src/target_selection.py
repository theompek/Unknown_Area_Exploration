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

    def selectTarget(self, init_ogm, coverage, robot_pose, origin, resolution, force_random=False):
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
            self.previous_target = self.selectRandomTarget(ogm, coverage, brush, ogm_limits)

        # Cost based target selection
        if self.method == 'costBasedTargetSelection' and force_random == False:

                nextTarget = self.costBasedTargetSelection(coverage, brush, robot_pose, resolution, origin, nodes)
                # Check if was found a target
                if nextTarget != False:
                    self.previous_target = nextTarget
                else:
                    Print.art_print("Select random target", Print.ORANGE)
                    self.previous_target = self.selectRandomTarget(ogm, coverage, brush, ogm_limits)

        return self.previous_target

    def costBasedTargetSelection(self, coverage, brush, robot_pose, resolution, origin, nodes):
        tinit = time.time()

        # Get the robot pose in pixels
        [rx, ry] = [int(round(robot_pose['x_px'] - origin['x'] / resolution)), int(round(robot_pose['y_px'] - origin['y'] / resolution))]

        node_cost = []
        target_flag = False
        #Find the cost for all nodes
        for current_node, node in enumerate(nodes):
            # Path sub_goals
            sub_goals = self.myflip(self.path_planning.createPath([rx, ry], node, resolution), 0)
            # Check if there is a path
            if sub_goals.shape[0] > 2:
                # Calculate parameters for the costs
                vectors = sub_goals[1:, :] - sub_goals[:-1, :]

                #Distances
                all_distances = np.sqrt(np.einsum('ij,ij->i', vectors, vectors))

                # Cosine of the angles
                angles_cosine = np.sum(vectors[1:, :] * vectors[:-1, :], axis=1) / np.linalg.norm(vectors[1:, :], axis=1) / np.linalg.norm(vectors[:-1, :], axis=1)

                #Coverage
                pathIndex = np.rint(sub_goals).astype(int)

                #Distance cost
                distances_cost = np.sum(all_distances)

                #Topological cost
                topological_cost = brush[node[0], node[1]]

                #Turn cost
                turn_cost = np.sum(abs(np.arccos(np.clip(angles_cosine, -1, 1))))

                #Calculate the coverage cost
                coverage_cost = 1 - np.sum(coverage[pathIndex[:, 0], pathIndex[:, 1]]) / (sub_goals.shape[0] * 255)

                #Save node's cost
                node_cost.append([current_node, distances_cost, topological_cost, turn_cost, coverage_cost])

                #We find at least one node so we rise the flag
                target_flag = True

        #We choose the best node
        if target_flag == True:
            node_cost = np.array(node_cost)

            # Normalize the cost
            node_cost[:, 1:] = 1 - ((node_cost[:, 1:] - np.min(node_cost[:, 1:], axis=0)) / \
                                 (np.max(node_cost[:, 1:], axis=0) - np.min(node_cost[:, 1:], axis=0)))

            # Calculatete the final cost
            final_cost = 8 * node_cost[:, 2] + 4 * node_cost[:, 1] + 2 * node_cost[:, 4] + node_cost[:, 3]

            # Find the next target
            index = int(node_cost[max(xrange(len(final_cost)), key=final_cost.__getitem__)][0])
            target = nodes[index]

            # The new target have to be a little farther than the previous one otherwise choose random
            distance = math.hypot(target[0] - self.previous_target[0], target[1] - self.previous_target[1])
            if distance < 10:
                if index < len(node_cost)-1:
                    target = nodes[index+1]
                else:
                    target = nodes[index-1]

            Print.art_print("Cost based selection Time: " + str(time.time() - tinit), Print.ORANGE)
            return target
        else:
            Print.art_print("Cost based selection failed **\_(*-*)_/**")
            return False

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


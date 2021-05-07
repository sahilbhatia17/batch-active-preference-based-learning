"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""
# preliminaries
#%matplotlib inline
import random
import time
import math
import numpy as np
import pdb
from scipy import stats
import scipy.spatial
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from IPython import display

class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start, goal, blue_obstacle_list, pink_obstacle_list, purple_obstacle_list, obstacle_list, rand_area,
                 expand_dis=2.0, path_resolution=0.5, goal_sample_rate=5, max_iter=500):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.blue_obstacle_list = blue_obstacle_list
        self.pink_obstacle_list = pink_obstacle_list
        self.purple_obstacle_list = purple_obstacle_list
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, feature_weights, feature_threshold, gt_weights, gt_threshold, simulator, mode="normal", animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """
        violations = 0
        self.node_list = [self.start]
        #pdb.set_trace()
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            if mode == "reward":
                nearest_ind = self.get_best_node_index(self.node_list, rnd_node, feature_weights)
            else:
                nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                if sum(simulator.compute_features_single(new_node.x, new_node.y) * gt_weights) < gt_threshold:
                    # we are violating the ground truth, check if we're violating what we learned
                    if mode == "rejection":
                        if sum(simulator.compute_features_single(new_node.x,
                                                                 new_node.y) * feature_weights) >= feature_threshold:
                            violations += 1
                            self.node_list.append(new_node)
                    else:
                        violations += 1
                        self.node_list.append(new_node)
                else:
                    self.node_list.append(new_node)


            if animation:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1), violations

                #self.node_list.append(new_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                            random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
    
        #commenting out the black obstacles for the preference learning tests for now
        for (ox, oy, size) in self.obstacle_list:
           self.plot_circle(ox, oy, size, color='black')
        #pdb.set_trace()

        for (ox, oy, size) in self.blue_obstacle_list:
            self.plot_circle(ox, oy, size)
        for (ox, oy, size) in self.pink_obstacle_list:
            self.plot_circle(ox, oy, size, color='pink')
        for (ox, oy, size) in self.purple_obstacle_list:
            self.plot_circle(ox, oy, size, color='purple')
        # for (ox, oy, size) in self.obstacle_list:
        #     self.plot_circle(ox, oy, size, color='b')

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        display.clear_output(wait=True)
        display.display(plt.gcf())

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def get_best_node_index(node_list, rnd_node, feature_weights, sim):
        dlist = np.array([-1 * ((node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2) for node in node_list])
        clist = np.array([sum(sim.compute_features_single(node.x, node.y) * feature_weights) for node in node_list])
        combined_list = dlist + clist
        minind = combined_list.index(max(combined_list))
        return minind


    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size ** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def get_obstacle_list():
    obstacle_list = [
        (5, 7, 1),
        (7, 7, 1),
        (9, 7, 1),
        (11, 7, 1),
        (3, 1, 1),
        (5, 10, 1),
        (10, 9, 1),
        (10, 12, 1)
    ]
    return obstacle_list


def plan_with_rrt(rrt, expand_dis=3.0, post_process_path=None, show_animation=False):
    # ====Search Path with RRT====
    obstacle_list = get_obstacle_list()
    rrt.expand_dis = expand_dis
    path = rrt.planning(animation=show_animation)

    processed_path = path
    if path is not None and post_process_path is not None:
        processed_path = post_process_path(path, rrt)

    if path is not None and show_animation:
        rrt.draw_graph()
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r', label='RRT')
        plt.grid(True)
        if post_process_path:
            plt.plot([x for (x, y) in processed_path], [y for (x, y) in processed_path],
                     color='cyan', label='RRT+post-processing')
            plt.legend()
    return processed_path

def run_rrt_comparison(gt_rewards, gt_threshold, learned_weights, learned_threshold, max_iter, mode='normal'):
    initial_state = [0, 0]
    expand_dis = 1.0
    goal = [15, 14]
    rand_area = [-2, 15]
    lower_bound = -2
    upper_bound = 15
    obstacle_list = get_obstacle_list()
    blue_obstacle_list = [(3, 6, 1), (3, 10, 1), (9, 5, 1),
                               (14, 7, 1), (12, 12, 1), (5, 13, 1),
                               (3, -1, 1), (5, -1, 1),
                               # added
                               (-1, 4, 1), (0, 8, 1)]
    pink_obstacle_list = [(8, 10, 1), (7, 5, 1), (10, 1, 1), (13, 10, 1), (15, 4, 1),
                               (14, 5, 1), (2, 4, 1), (5, 1, 1)]
    purple_obstacle_list = [(3, 8, 1), (5, 12, 1), (5, 5, 1), (12, 3, 1), (2, 7, 1), (-1, -1, 1)]
    #max_iter = 100
    rrt = RRT(initial_state, goal, blue_obstacle_list,
                                         pink_obstacle_list,
                                         purple_obstacle_list, obstacle_list, rand_area=rand_area,
                                         expand_dis=expand_dis, max_iter=max_iter)
    #pdb.set_trace()
    simulator_obj = CirclesSimulation("thing")
    path_tuple = rrt.planning(learned_weights, learned_threshold, gt_rewards, gt_threshold, simulator_obj, mode=mode, animation=False)
    if path_tuple is None:
        print("No path was found for RRT planning with method {}.".format(mode))
    else:
        path, violations = path_tuple
        print("NUMBER OF VIOLATIONS IS: {}".format(violations))
        simulator_obj.trajectory = path
        simulator_obj.watch()

    # rrt_obj = simulator_obj.rrt
    # rrt_obj.end = rrt_obj.Node(goal[0], goal[1])

class CirclesSimulation():
    def __init__(self, name, total_time=50, recording_time=[0, 50]):
        self.initial_state = [0, 0]
        self.expand_dis = 3.0
        self.goal = [15, 14]
        self.rand_area = [-2, 15]
        self.lower_bound = -2
        self.upper_bound = 15
        self.blue_obstacle_list = [(3, 6, 1), (3, 10, 1), (9, 5, 1),
                                   (14, 7, 1), (12, 12, 1), (5, 13, 1),
                                   (3, -1, 1), (5, -1, 1),
                                   # added
                                   (-1, 4, 1), (0, 8, 1)]
        self.pink_obstacle_list = [(8, 10, 1), (7, 5, 1), (10, 1, 1), (13, 10, 1), (15, 4, 1),
                                   (14, 5, 1), (2, 4, 1), (5, 1, 1)]
        self.purple_obstacle_list = [(3, 8, 1), (5, 12, 1), (5, 5, 1), (12, 3, 1), (2, 7, 1), (-1, -1, 1)]
        self.max_iter = 100
        self.rrt = RRT(self.initial_state, self.goal, self.blue_obstacle_list,
                                             self.pink_obstacle_list,
                                             self.purple_obstacle_list, get_obstacle_list(), rand_area=self.rand_area,
                                             expand_dis=self.expand_dis, max_iter=self.max_iter)

        self.current_position = self.initial_state
        self.input_size = 2
        self.trajectory = []
        self.reset()
        self.viewer = None

    def compute_min_distance(self, xcoord, ycoord, obs_list):
        return min((xcoord - obs[0]) ** 2 + (ycoord - obs[1]) ** 2 for obs in obs_list)

    def compute_feature_tolerance(self, xcoord, ycoord, obs_list):
        feature_value = 0
        # the bigger this value, the more "regions" we have entered / more deeply
        for obs in obs_list:
            obsx, obsy, size = obs
            distance_from_center = (xcoord - obsx) ** 2 + (ycoord - obsy) ** 2
            if distance_from_center < size:
                feature_value += (size - distance_from_center)
        return feature_value

    def compute_features_single(self, xcoord, ycoord):
        blue_feature_value = self.compute_feature_tolerance(xcoord, ycoord, self.blue_obstacle_list)
        pink_feature_value = self.compute_feature_tolerance(xcoord, ycoord, self.pink_obstacle_list)
        purple_feature_value = self.compute_feature_tolerance(xcoord, ycoord, self.purple_obstacle_list)
        # the bigger this value, the more "regions" we have entered / more deeply

        return np.array([blue_feature_value, pink_feature_value, purple_feature_value])

    def get_features_over_trajectory(self, trajectory):
        # compute minimum distance from each obstacle in the list:
        feature_list = np.zeros((len(trajectory), 3))
        for idx in range(len(trajectory)):
            # x, y coordinates
            xcoord, ycoord = trajectory[idx]
            # now, go through each obstacle list and compute the min distance
            blue_feat = self.compute_feature_tolerance(xcoord, ycoord, self.blue_obstacle_list)
            pink_feat = self.compute_feature_tolerance(xcoord, ycoord, self.pink_obstacle_list)
            purple_feat = self.compute_feature_tolerance(xcoord, ycoord, self.purple_obstacle_list)
            feature_list[idx] = [blue_feat, pink_feat, purple_feat]
        return feature_list

    def get_features_full(self):
        return self.get_features_over_trajectory(self.trajectory)

    def reset(self):
        #super(CirclesSimulation, self).reset()
        self.current_position = [0, 0]
        self.trajectory = []

    def initialize_positions(self):
        self.current_position = [0, 0]
        self.trajectory = []
        self.trajectory.append(self.current_position[:])

    def run(self, reset=False):
        if reset:
            self.reset()
        else:
            self.initialize_positions()

        #        print(self.trajectory)
        #        print(self.ctrl_array)
        for i in range(self.total_time):
            self.current_position[0] += self.ctrl_array[i][0]
            # set boundaries so we don't go off screen
            self.current_position[0] = min(max(self.lower_bound, self.current_position[0]), self.upper_bound)
            self.current_position[1] += self.ctrl_array[i][1]
            self.current_position[1] = min(max(self.lower_bound, self.current_position[1]), self.upper_bound)
            self.trajectory.append(self.current_position[:])

        self.alreadyRun = True

    # I keep all_info variable for the compatibility with mujoco wrapper
    def get_trajectory(self, all_info=True):
        if not self.alreadyRun:
            self.run()
        return self.trajectory.copy()

    def get_recording(self, all_info=True):
        traj = self.get_trajectory(all_info=all_info)
        return traj

    def watch(self, repeat_count=1):
        self.rrt.draw_graph()
        #        print(self.trajectory)
        plt.plot([x for (x, y) in self.trajectory], [y for (x, y) in self.trajectory], '-r', label='trajectory')
        plt.grid(True)
        plt.legend()
        plt.show(block=False)
        # plt.pause(5)
        # plt.close()




# class RRTStar(RRT):
#     """
#     Class for RRT Star planning
#     """
#
#     class Node(RRT.Node):
#         def __init__(self, x, y):
#             super().__init__(x, y)
#             self.cost = 0.0
#
#     def __init__(self, start, goal, blue_obstacle_list, pink_obstacle_list, purple_obstacle_list, rand_area,
#                  expand_dis=30.0,
#                  path_resolution=1.0,
#                  goal_sample_rate=20,
#                  max_iter=300,
#                  connect_circle_dist=50.0
#                  ):
#         super().__init__(start, goal, blue_obstacle_list, pink_obstacle_list, purple_obstacle_list,
#                          rand_area, expand_dis, path_resolution, goal_sample_rate, max_iter)
#         """
#         Setting Parameter
#         start:Start Position [x,y]
#         goal:Goal Position [x,y]
#         obstacleList:obstacle Positions [[x,y,size],...]
#         randArea:Random Sampling Area [min,max]
#         """
#         self.connect_circle_dist = connect_circle_dist
#         self.goal_node = self.Node(goal[0], goal[1])
#
#     def planning(self, animation=True, search_until_max_iter=True):
#         """
#         rrt star path planning
#         animation: flag for animation on or off
#         search_until_max_iter: search until max iteration for path improving or not
#         """
#
#         self.node_list = [self.start]
#         for i in range(self.max_iter):
#             rnd = self.get_random_node()
#             nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
#             new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
#
#             if self.check_collision(new_node, self.obstacle_list):
#                 near_inds = self.find_near_nodes(new_node)
#                 new_node = self.choose_parent(new_node, near_inds)
#                 if new_node:
#                     self.node_list.append(new_node)
#                     self.rewire(new_node, near_inds)
#
#             if animation and i % 5 == 0:
#                 self.draw_graph(rnd)
#
#             if (not search_until_max_iter) and new_node:  # check reaching the goal
#                 last_index = self.search_best_goal_node()
#                 if last_index:
#                     return self.generate_final_course(last_index)
#
#         last_index = self.search_best_goal_node()
#         if last_index:
#             return self.generate_final_course(last_index)
#
#         return None
#
#     def choose_parent(self, new_node, near_inds):
#         if not near_inds:
#             return None
#
#         # search nearest cost in near_inds
#         costs = []
#         for i in near_inds:
#             near_node = self.node_list[i]
#             t_node = self.steer(near_node, new_node)
#             if t_node and self.check_collision(t_node, self.obstacle_list):
#                 costs.append(self.calc_new_cost(near_node, new_node))
#             else:
#                 costs.append(float("inf"))  # the cost of collision node
#         min_cost = min(costs)
#
#         if min_cost == float("inf"):
#             print("There is no good path.(min_cost is inf)")
#             return None
#
#         min_ind = near_inds[costs.index(min_cost)]
#         new_node = self.steer(self.node_list[min_ind], new_node)
#         new_node.parent = self.node_list[min_ind]
#         new_node.cost = min_cost
#
#         return new_node
#
#     def search_best_goal_node(self):
#         dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
#         goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]
#
#         safe_goal_inds = []
#         for goal_ind in goal_inds:
#             t_node = self.steer(self.node_list[goal_ind], self.goal_node)
#             if self.check_collision(t_node, self.obstacle_list):
#                 safe_goal_inds.append(goal_ind)
#
#         if not safe_goal_inds:
#             return None
#
#         min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
#         for i in safe_goal_inds:
#             if self.node_list[i].cost == min_cost:
#                 return i
#
#         return None
#
#     def find_near_nodes(self, new_node):
#         nnode = len(self.node_list) + 1
#         r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
#         # if expand_dist exists, search vertices in a range no more than expand_dist
#         if hasattr(self, 'expand_dis'):
#             r = min(r, self.expand_dis)
#         dist_list = [(node.x - new_node.x) ** 2 +
#                      (node.y - new_node.y) ** 2 for node in self.node_list]
#         near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
#         return near_inds
#
#     def rewire(self, new_node, near_inds):
#         for i in near_inds:
#             near_node = self.node_list[i]
#             edge_node = self.steer(new_node, near_node)
#             if not edge_node:
#                 continue
#             edge_node.cost = self.calc_new_cost(new_node, near_node)
#
#             no_collision = self.check_collision(edge_node, self.obstacle_list)
#             improved_cost = near_node.cost > edge_node.cost
#
#             if no_collision and improved_cost:
#                 self.node_list[i] = edge_node
#                 self.propagate_cost_to_leaves(new_node)
#
#     def calc_new_cost(self, from_node, to_node):
#         d, _ = self.calc_distance_and_angle(from_node, to_node)
#         return from_node.cost + d
#
#     def propagate_cost_to_leaves(self, parent_node):
#
#         for node in self.node_list:
#             if node.parent == parent_node:
#                 node.cost = self.calc_new_cost(parent_node, node)
#                 self.propagate_cost_to_leaves(node)
#
# def plan_with_rrt_star(gx=6.0, gy=10.0, expand_dis=30, post_process_path=None, show_animation=False):
#     # ====Search Path with RRT====
#     obstacleList = [
#         (5, 5, 1),
#         (3, 6, 2),
#         (3, 8, 2),
#         (3, 10, 2),
#         (7, 5, 2),
#         (9, 5, 2),
#         (8, 10, 1)
#     ]  # [x, y, radius]
#     # Set Initial parameters
#     rrt_star = RRTStar(start=[0, 0],
#               expand_dis=30,
#               goal=[gx, gy],
#               rand_area=[-2, 15],
#               obstacle_list=obstacleList)
#     path = rrt_star.planning(animation=show_animation)
#
#     if path is not None and show_animation:
#             display.display(plt.gcf())
#             rrt_star.draw_graph()
#             plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
#             plt.grid(True)
#             display.clear_output(wait=True)
#     return path

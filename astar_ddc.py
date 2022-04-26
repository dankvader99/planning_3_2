import cv2
import numpy as np
from math import dist
import matplotlib.pyplot as plt


class Node:
    def __init__(self, pos, theta, parent, action, cost, cost2go = 0):
        self.pos = pos
        self.theta = theta
        self.parent = parent
        self.cost = cost
        self.cost2go = cost2go
        self.action = action

    def __lt__(self, other):
        return self.cost + self.cost2go < other.cost + other.cost2go
    
    @property
    def key(self):
        return int(self.pos[0]), int(self.pos[1]), int(self.theta/10)


def check_goal(node, goal):
    # print(node.pos, goal.pos)
    dt = dist(node.pos, goal.pos)     
    return dt < 30        

       
def A_Star(graph, start_node, goal_node):
    ''' Search for a path from start to given goal position
    '''
    # print(start_node.pos, goal_node.pos)
    explored = {start_node.key : start_node}
    cost_dict = {start_node.key: dist(start_node.pos, goal_node.pos)}
    # Run A_Star
    while len(cost_dict):
        key = min(cost_dict, key = cost_dict.get)
        cost_dict.pop(key)
        curr_node = explored[key]
        graph.close(key)

        # if the new_node is goal backtrack and return path
        if check_goal(curr_node, goal_node):
            return backtrack(graph, curr_node), explored

        # get all children of the current node
        for action in graph.actions:
            child_node, new_cost, _ = graph.do_action(curr_node, action)

              # if the new position is not free, skip the below steps
            if not graph.is_free(child_node.key): 
                continue

            if child_node.key in explored:
                child_node = explored[child_node.key]

            # update cost and modify cost_dict
            if new_cost < cost_dict.get(child_node.key, np.inf):
                child_node.parent = curr_node
                child_node.cost = new_cost 
                child_node.action = action
                cost_dict[child_node.key] = new_cost + 2*dist(child_node.pos, goal_node.pos)
                explored[child_node.key] = child_node
        
    # return None if no path is found.
    return None, explored

class Graph:

    actions = None

    def __init__(self):
        # size of the map
        self.size = (500, 500)
        self.cspace = np.zeros((500, 500, 36), np.uint8)
    
    def is_obs(self, pt, cl = 15):
        x, y, th = pt

        if np.sqrt((x - 100)**2 + (y - 100)**2) <= 50 + cl:
            return 1
        if np.sqrt((x - 100)**2 + (y - 400)**2) <= 50 + cl:
            return 1

        if 13 - cl <= x  and x <= 86 + cl:
            if 213 - cl <= y and y <= 288 + cl:
                return 1

        if 188 - cl <= x and x <= 313 + cl:
            if 213 - cl <= y and y <= 288 + cl:
                return 1
        
        if 363 - cl <= x and x <= 438 + cl:
            if 100 - cl <= y and y <= 200 + cl:
                return 1
        return 0
            
    def is_free(self, pos):
        # inside the map size
        if pos[0] < 0 or pos[1] < 0: 
            return False 
        if pos[0] >= self.size[0] or pos[1] >= self.size[1]:  
            return False 
        # and node is not closed or obstacle
        return self.is_obs(pos) == 0 and self.cspace[pos] == 0 
    
    def close(self, pos):
        self.cspace[pos] = 1
    
    def do_action(self, node, action):
        # add the action step to node position to get new position
        x, y = node.pos
        theta = node.theta*np.pi/180
        UL, UR = action

        r = 0.038*100
        L = 0.354*100

        dt = 0.1
        xs, ys = [], []

        # UL, UR = UL*0.10472, UR*0.10472
        v = 0.5*r * (UL + UR) * dt
        w = ((r / L) * (UR - UL) * dt)
        D = 0
        for _ in range(10):
            x += v * np.cos(theta)
            y += v * np.sin(theta)
            theta += w
            D = D + v
            xs.append(x)
            ys.append(y)

        pos = int(x), int(y)
        theta = int(((theta*180/np.pi)%360)/10)*10
        new_cost = node.cost + D

        child_node = Node(pos, theta, node, action, new_cost, cost2go = 0)

        return child_node, new_cost, (xs, ys)
 
    
    def get_mapimage(self):
        img = np.full(self.size, 255, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.circle(img, (100, 100), 50, (0, 0, 0), thickness = -1)
        cv2.circle(img, (100, 400), 50, (0, 0, 0), thickness = -1)

        cv2.rectangle(img, (13, 213), (86, 288), (0, 0, 0), thickness= -1)
        cv2.rectangle(img, (188, 213), (313, 288), (0, 0, 0), thickness= -1)
        cv2.rectangle(img, (363, 400), (438, 300), (0, 0, 0), thickness= -1)
        return img


def backtrack(graph, node):
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    return path
    

def visualize(graph, path, explored):
    ''' Visualise the exploration and the recently found path
    '''
    img = graph.get_mapimage()

    track = path 
    h, w, _ = img.shape
    out = cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 60.0, (w, h))
    
    i = 0
    for key, node in explored.items():
        parent = node.parent
        if parent is None: continue
        _, _, (xs, ys) = graph.do_action(parent, node.action)
        cv2.polylines(img, [np.int32(np.c_[xs, w - np.array(ys) - 1])], False, [0, 80 ,0], 1)
        if i%2 == 0:
            out.write(img)
            cv2.imshow('hi', img)
            cv2.waitKey(1)
        i += 1
        
    for node in path[1:]:
        parent = node.parent
        _, _, (xs, ys) = graph.do_action(parent, node.action)
        cv2.polylines(img, [np.int32(np.c_[xs, w - np.array(ys) - 1])], False, [0, 0, 255], 2)

    out.release()
    cv2.imshow('hi', img)
    cv2.waitKey(0)
           

# when running as main 
if __name__ == '__main__':
    #give start and goal states and orientation and robot radius & clearance 

    
    x_s, y_s, theta = 50, 50, 0
    x_g, y_g = 450, 450

    rpm1, rpm2 = 25, 50

    start_node = Node((x_s, y_s), theta, None, None, 0)
    goal_node = Node((x_g, y_g), 0, None, None, 0)

    graph = Graph()

    graph.actions = np.array([(0, rpm1), (rpm1, 0), (rpm1, rpm1), 
                              (0, rpm2), (rpm2, 0), (rpm2, rpm2), 
                              (rpm1, rpm2), (rpm2, rpm1)])*0.10472

    path, explored = A_Star(graph, start_node, goal_node)

    img = graph.get_mapimage()

    visualize(graph, path, explored)
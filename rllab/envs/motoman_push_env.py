from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np

import os
import socket
g_repo_path = os.environ['PHYSICS_SIMULATOR_PATH']

# Input Pose File path
init_pose_file = g_repo_path + '/tmp/init_pose.txt'
final_pose_file = g_repo_path + '/tmp/final_pose.txt'
action_file = g_repo_path + '/tmp/action.txt'
# -0.107687, -0.535722, 0.563821, 0.941487, -0.0294104, 0.335604, 0.0103682

class MOtomanPushEnv(Env):
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(7,))

    @property
    def action_space(self):
        return Box(low=0.05, high=1.0, shape=(1,))

    def reset(self):
        self._state = [-0.107687, -0.535722, 0.791, 0.0103682, 0.941487, -0.0294104, 0.335604] #q:wxyz #np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        # self._state = self._state + action
        # x, y = self._state
        # reward = - (x ** 2 + y ** 2) ** 0.5
        print ('\n init state and action: ', self._state, action)
        mass = 0.24351668  
        friction = 0.05537921
        obj_goal = [self._state[0]+1.0, self._state[1], self._state[2]]


        with open(init_pose_file, "w+") as file:
            file.write("%f %f %f %f %f %f %f %f %f %f %f %f \n" % \
                       (mass, friction, self._state[0], self._state[1], self._state[3], self._state[4], \
                        self._state[5], self._state[6], action, self._state[0], obj_goal[0], obj_goal[1]))
     
        # Create a TCP/IP socket
        sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Connect the socket to the port where the server is listening
        serv_address = ('localhost', 50007)
        print('waiting for a connection')
        sock2.connect(serv_address)
        print('Connected')
        sock2.sendall(b'hi')
        sock2.close()   

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind the socket to the port
        server_address = ('localhost', 10022)
        sock.bind(server_address)

        # Listen for incoming connections
        sock.listen(1)        
        connection, client_address = sock.accept()

        # Receive the data in small chunks and retransmit it
        data = connection.recv(16)
        connection.close()
        sock.close()             

        Y_new = np.zeros((7, ))
        with open(final_pose_file, "r") as file:
            for line in file:
                vals = line.split()
                for i in range(0,7):
                    Y_new[i] = vals[i]
        self._state = Y_new
        next_observation = np.copy(self._state)

        done = True #abs(x) < 0.01 and abs(y) < 0.01

        # q_dist = np.abs(1 - np.sum(np.multiply(rotation, rotation_gt)**2))
        dist = (( Y_new[0]-obj_goal[0])**2+( Y_new[1]-obj_goal[1])**2 + ( Y_new[2]-obj_goal[2])**2) ** 0.5

        k = 2
        reward = np.e**(- k*dist)


        print ('action: ', action, 'obj_goal: ', obj_goal, 'Y_new: ', Y_new[0:3], 'dist: ', dist, 'reward: ', reward)

        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

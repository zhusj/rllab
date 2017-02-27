from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np

import os
import socket

# communication with PRACSYS
from prx_simulation.msg import *

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool

import geometry_msgs.msg


g_repo_path = os.environ['PHYSICS_SIMULATOR_PATH']

# Input Pose File path
init_pose_file = g_repo_path + '/tmp/init_pose.txt'
final_pose_file = g_repo_path + '/tmp/final_pose.txt'
action_file = g_repo_path + '/tmp/action.txt'
# -0.107687, -0.535722, 0.563821, 0.941487, -0.0294104, 0.335604, 0.0103682
rospy.init_node('rllab', anonymous=True)


def reinit_callback(data):
    global flag
    flag = data.data
    print ("Reinitialize requested from PRACSYS, flag: ", flag)
    predict()

def obj_pose_callback(data):
    global object_pose, object_name
    object_name = data.object_names
    object_pose = data.object_poses
    # print ("\n received object name/pose:"+object_name+" : "+str(object_pose))

def ee_pose_callback(data):
    global ee_pose, ee_name
    ee_name = data.object_names
    ee_pose = data.object_poses
    # print ("\n received ee name/pose:"+ee_name+" : "+str(ee_pose))

def obj_final_pose_callback(data):
    global obj_final_pose
    obj_final_pose = np.zeros((7,))
    # print ('obj_final_pose: ', obj_final_pose)
    obj_final_pose[0] = data.pose.position.x
    obj_final_pose[1] = data.pose.position.y
    obj_final_pose[2] = data.pose.position.z
    obj_final_pose[3] = data.pose.orientation.x
    obj_final_pose[4] = data.pose.orientation.y
    obj_final_pose[5] = data.pose.orientation.z
    obj_final_pose[6] = data.pose.orientation.w
    # print ("\n received obj_final_pose: ", obj_final_pose)

class MOtomanPushEnvReal(Env):
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(7,))

    @property
    def action_space(self):
        return Box(low=0.1, high=1.0, shape=(1,))

    def reset(self):
        rospy.Subscriber('/physics_simulator_obj_node', object_name_pose_message, obj_pose_callback)
        rospy.wait_for_message("physics_simulator_obj_node", object_name_pose_message )
        # print ('object_pose: ', object_pose[0], object_pose[1],object_pose[2], object_pose[6], object_pose[3], \
        #             object_pose[4], object_pose[5])
        self._state = [ object_pose[0], object_pose[1],object_pose[2], object_pose[6], object_pose[3], \
                    object_pose[4], object_pose[5]] #q:wxyz #np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        rospy.Subscriber('/baxter_detected_obj_pose', PoseStamped, obj_final_pose_callback)

        obj_goal = [self._state[0], self._state[1]+1.0, self._state[2]]

        print ('\n init state, obj_goal and action: ', self._state, obj_goal, action)

        pub = rospy.Publisher('action', PoseStamped, queue_size=10, latch =True)

        # rospy.Subscriber('/physics_simulator_request_init', Bool, reinit_callback)

        # rospy.Subscriber('/physics_simulator_ee_node', object_name_pose_message, ee_pose_callback)

        # rate = rospy.Rate(10.0)

    # while not rospy.is_shutdown():
        pose = geometry_msgs.msg.PoseStamped()        
        pose.pose.position.x = float(action)
        pose.pose.position.y = 2 #//anything greater than 1
        pose.pose.position.z = 0
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.z = 0
        pose.pose.orientation.w = 1
        pub.publish(pose)

        # rospy.loginfo(pose)

        rospy.wait_for_message("baxter_detected_obj_pose", PoseStamped )



        done = True #abs(x) < 0.01 and abs(y) < 0.01


        # obj pos transform from baxter(bx,by) to motoman(mx,my)
        # mx = 1.4726 + by, my = 0.8322-bx
        Y_new = obj_final_pose
        # Y_new_tmp = Y_new
        # print ('Y_new: ', Y_new)

        if obj_final_pose[0]==10:
            Y_new[2] = -10
        else:
            Y_new[2] = obj_goal[2]
        # print ('Y_new: ', Y_new)

        # # print ('obj_final_pose_tmp: ', obj_final_pose_tmp)
        # Y_new[0] = 1.4726 + Y_new_tmp[1]
        # print ('obj_final_pose: ', obj_final_pose)

        # # print ('obj_final_pose_tmp: ', obj_final_pose_tmp)
        # print ('Y_new: ', Y_new)
        # Y_new[1] = 0.8322 - Y_new_tmp[0]
        # print ('Y_new: ', Y_new)

        # q_dist = np.abs(1 - np.sum(np.multiply(rotation, rotation_gt)**2))
        dist = (( Y_new[0]-obj_goal[0])**2+( Y_new[1]-obj_goal[1])**2 + ( Y_new[2]-obj_goal[2])**2) ** 0.5

        k = 2
        reward = np.e**(- k*dist)


        print ('action: ', action, 'obj_goal: ', obj_goal, 'Y_new: ', Y_new[0:3], 'dist: ', dist, 'reward: ', reward)
        
        self._state = Y_new
        next_observation = np.copy(self._state)
        # rate.sleep()

        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

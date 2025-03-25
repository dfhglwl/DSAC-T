import numpy as np
import math

class Quadrotor:
    def __init__(self,  g=9.8):
        # dyna para
        self.g = g
        self.Jx = 1
        self.Jy = 1
        self.Jz = 1
        self.m = 1
        self.c = 0.01
        self.l = 0.4
        self.J_B = np.diag(np.array([self.Jx,self.Jy,self.Jz]))
        self.g_I = np.array([0, 0, -self.g])
        self.dt = 0.1

        # reward para
        self.wr = 0.02
        self.wv = 0.001
        self.wq = 0.002
        self.ww = 0.001

    def reset(self):

        #self.state = np.random.randint(-10,10,size=[13,])+np.random.random((13,))
        r_state = np.random.randint(-10, 10, size=[3, ]) + np.random.random((3,))
        v_state = np.random.randint(-1, 1, size=[3, ]) + np.random.random((3,))
        q_state = np.random.randint(-1, 1, size=[4, ]) + np.random.random((4,))
        w_state = np.random.randint(-1, 1, size=[3, ]) + np.random.random((3,))
        self.state = np.hstack([r_state, v_state, q_state, w_state])
        #print(state, state.shape)

        return self.state

    def step(self, state, action, wthrust=0.0001):
        rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = state # state is a np.array
        f1, f2, f3, f4 = action
        self.r_I = np.array([rx, ry, rz])
        self.v_I = np.array([vx, vy, vz])
        self.q = np.array([q0, q1, q2, q3])
        self.w_B = np.array([wx, wy, wz])
        self.T_B = np.array([f1, f2, f3, f4])
        thrust = (self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3])
        self.thrust_B = np.array([0, 0, thrust])

        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c
        self.M_B = np.array([Mx, My, Mz])

        C_B_I = self.dir_cosine(self.q)  # inertial to body
        C_I_B = np.transpose(C_B_I)

        # dyna
        dr_I = self.v_I
        dv_I = 1 / self.m * np.dot(C_I_B, self.thrust_B) + self.g_I
        dq = 1 / 2 * np.dot(self.omega(self.w_B), self.q)
        dw = np.dot(np.transpose(self.J_B), self.M_B - np.dot(np.dot(self.skew(self.w_B), self.J_B), self.w_B))

        self.X = np.hstack((self.r_I, self.v_I, self.q, self.w_B))
        self.U = self.T_B
        self.f = np.hstack((dr_I, dv_I, dq, dw))
        new_state = state + (self.f * self.dt)
        # UAV_low = np.array([-1000, -1000, -1000, -5, -5, -5, -10, -10, -10, -10, -10, -10, -10])
        # UAV_up = np.array([1000, 1000, 1000, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10])
        new_state1 = new_state # np.clip( new_state, UAV_low, UAV_up)

        # reward # goal position in the world frame
        goal_r_I = np.array([0, 0, 0])
        self.cost_r_I = np.dot(self.r_I - goal_r_I, self.r_I - goal_r_I)     #TRY
        # goal velocity
        goal_v_I = np.array([0, 0, 0])
        self.cost_v_I = np.dot(self.v_I - goal_v_I, self.v_I - goal_v_I)
        # final attitude error
        goal_q = self.toQuaternion(0, [0, 0, 1])
        goal_R_B_I = self.dir_cosine(goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q = np.trace(np.identity(3) - np.dot(np.transpose(goal_R_B_I), R_B_I))
        # auglar velocity cost
        goal_w_B = np.array([0, 0, 0])
        self.cost_w_B = np.dot(self.w_B - goal_w_B, self.w_B - goal_w_B)
        # the thrust cost
        self.cost_thrust = np.dot(self.T_B, self.T_B)

        # self.wr = 0.01
        # self.wv = 0.001
        # self.wq = 0.001
        # self.ww = 0.001
        self.path_cost = self.wr * self.cost_r_I + \
                          self.wv * self.cost_v_I + \
                          self.ww * self.cost_w_B + \
                          self.wq * self.cost_q + \
                           wthrust * self.cost_thrust

        reward =  self.path_cost

        if self.cost_r_I <= 9 :         # 到达目的地
            reward = reward - 100

        elif  self.cost_r_I >= 350 :    # 距离偏离目的地
            reward = reward + 200

        # if    np.all(dv_I*self.r_I < 0): # 加速度和位移方向相反
        #     reward = reward - 3



        # done
        if self.cost_r_I >= 350 :
            done = True

        elif self.cost_r_I <= 9:
             done = True

        # elif   self.cost_r_I <= 250 and not np.all(dv_I*self.r_I > 0): # 加速度和位移方向相反
        #      done = True
        #      print('velocity is not same as force')

        else :
            done = False


        return new_state1, - reward, done, self.r_I, self.cost_r_I


    def dir_cosine(self, q):
        C_B_I = np.array([
            [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
            [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
            [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)] ])
        # C_B_I = np.array([
        #     [1 - 2 * (q[2] ** 2 + q[3] ** 2),2 * (q[1] * q[2] - q[0] * q[3]),2 * (q[1] * q[3] + q[0] * q[2])],
        #     [2 * (q[1] * q[2] + q[0] * q[3]),1 - 2 * (q[1] ** 2 + q[3] ** 2),2 * (q[2] * q[3] - q[0] * q[1])],
        #     [2 * (q[1] * q[3] - q[0] * q[2]),2 * (q[2] * q[3] + q[0] * q[1]),1 - 2 * (q[1] ** 2 + q[2] ** 2)]])
        return C_B_I

    def omega(self, w):
        omeg = np.array([[0, -w[0], -w[1], -w[2]],
                         [w[0], 0, w[2], -w[1]],
                         [w[1], -w[2], 0, w[0]],
                         [w[2], w[1], -w[0], 0]])
        return omeg

    def skew(self, v):
        v_cross = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])

        return v_cross

    def toQuaternion(self, angle, dir):
        if type(dir) == list:
            dir = np.array(dir)
        dir = dir / (np.linalg.norm(dir)+0.00001)
        quat = np.zeros(4)
        quat[0] = math.cos(angle / 2)
        quat[1:] = math.sin(angle / 2) * dir
        return quat.tolist()

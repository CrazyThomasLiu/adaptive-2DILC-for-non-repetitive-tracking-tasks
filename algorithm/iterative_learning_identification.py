import pdb
import typing
import numpy as np
import copy
import pandas as pd
import csv

"MIMO Iterative Learning Identification for the time-varying batch systems"
class ILI:

    def __init__(self, batch_length: int, A_hat_t, B_hat_t, C_hat_t, Q_bar_t_1, R_bar_t_1,Q_bar_t_2, R_bar_t_2):
        self.batch_length=batch_length
        # the initial state space matrix
        self.A_hat_t = A_hat_t
        self.B_hat_t = B_hat_t
        self.C_hat_t = C_hat_t
        # the dimensions of the state space
        self.n=self.A_hat_t[0].shape[0] # the dimensions of the state variable
        self.m=self.B_hat_t[0].shape[1]   # the dimensions of the input variable
        self.q=self.C_hat_t[0].shape[0]  # the dimensions of the output variable
        # the cost function
        self.Q_bar_t_1 = Q_bar_t_1
        self.R_bar_t_1 = R_bar_t_1
        self.Q_bar_t_2 = Q_bar_t_2
        self.R_bar_t_2 = R_bar_t_2

        # define the matrix Phi [time, n, n+q]
        self.Phi_1 = np.concatenate((self.A_hat_t, self.B_hat_t), axis=2)
        self.Phi_2 = copy.deepcopy(self.C_hat_t)

        #self.Phi_itr=copy.deepcopy(self.Phi)
        #pdb.set_trace()
        #a=2


    def iterative_model(self,delta,r,x,y):
        z_bar_1=copy.deepcopy(delta[1:,:])
        #pdb.set_trace()
        z_1=np.concatenate((delta[0:self.batch_length,:], r), axis=1)

        #pdb.set_trace()
        for time in range(self.batch_length):
            # (R+zQz^{T})^{-1}

            #pdb.set_trace()
            tem=self.R_bar_t_1[time]+z_1[time].reshape((self.n+self.m,1))@self.Q_bar_t_1[time]@z_1[time].reshape((self.n+self.m,1)).T
            tem=np.linalg.inv(tem)

            for idx in range(self.n):
                phi_hat_1=tem@self.R_bar_t_1[time]@self.Phi_1[time,idx,:].reshape((self.n+self.m,1))+tem@z_1[time].reshape((self.n+self.m,1))@self.Q_bar_t_1[time]@z_bar_1[time,idx].reshape((1,1))

                self.Phi_1[time,idx,:]=phi_hat_1.reshape((self.n+self.m))
                #pdb.set_trace()
                #a=2

        z_bar_2 = copy.deepcopy(y[1:, :])
        # pdb.set_trace()
        z_2 = copy.deepcopy(x[1:, :])

        for time in range(self.batch_length):
            # (R+zQz^{T})^{-1}

            #pdb.set_trace()
            tem=self.R_bar_t_2[time]+z_2[time].reshape((self.n,1))@self.Q_bar_t_2[time]@z_2[time].reshape((self.n,1)).T
            tem=np.linalg.inv(tem)


            #pdb.set_trace()
            for idx in range(self.q):
                phi_hat_2=tem@self.R_bar_t_2[time]@self.Phi_2[time+1,idx,:].reshape((self.n,1))+tem@z_2[time].reshape((self.n,1))@self.Q_bar_t_2[time]@z_bar_2[time,idx].reshape((1,1))

                self.Phi_2[time+1,idx,:]=phi_hat_2.reshape((self.n))


                #pdb.set_trace()
                #a=2
        self.A_hat_t=self.Phi_1[:,:,0:self.n]
        self.B_hat_t=self.Phi_1[:,:,self.n:]
        self.C_hat_t=self.Phi_2

        return self.A_hat_t,self.B_hat_t,self.C_hat_t
        #pdb.set_trace()
        #a=2

import os
from control.matlab import *  # MATLAB-like functions
import pdb
import numpy as np
import control
import copy
import typing
import pprint
import pandas as pd

class  BatchSys:
    def __init__(self,batch_length:int,sample_time,sys,x_k0,A_t,B_t,C_t,y_ref,omega_t):
        self.batch_length=batch_length
        # the state space
        self.A_t = A_t
        self.B_t = B_t
        self.C_t = C_t
        self.sys=sys
        self.x_k0=copy.deepcopy(x_k0)   # k=1,2,3.... \infinity
        self.T=sample_time
        self.T_in=np.array((0.0,0.0))
        self.y_ref=y_ref
        self.y_ref_last=copy.deepcopy(self.y_ref)
        self.w=np.zeros((self.batch_length+1,self.C_t.shape[1]))
        self.omega_t = omega_t
        self.X0=self.x_k0[0:2]
        # the current batch state information
        self.x_batch =90*np.ones((self.batch_length+1,self.A_t.shape[1]))
        self.delta = np.zeros((self.x_batch.shape[0],self.x_batch.shape[1]))
        # the output error for the last batch
        self.e = self.y_ref -self.y_ref

        # the control signal
        self.control_signal=np.zeros((self.batch_length,self.B_t.shape[2]))
    def reset(self,y_ref):
        self.y_ref_last=copy.deepcopy(self.y_ref)
        self.y_ref=y_ref
        self.w=self.y_ref-self.y_ref_last+self.e
        self.T_in = np.array((0.0, 0.0))
        self.X0 = copy.deepcopy(self.x_k0[0:2])
        self.delta[0,:]=self.X0-self.x_batch[0,:]
        self.x_batch[0, :]=copy.deepcopy(self.X0)
        # compute the initial output error
        tem=self.C_t[0]@self.x_k0[0:2]
        self.e[0,:]=self.y_ref[0,:]-tem
        # construct the ILI data
        ILI_data=[self.delta[0,:],self.X0,tem]
        return np.hstack((self.delta[0,:],self.w[1:,:].flatten())),tem,self.e[0,:], ILI_data # control state vector, output, error
    def reset_randomly(self,y_ref,x_k0):
        self.y_ref_last=copy.deepcopy(self.y_ref)
        self.y_ref=y_ref
        self.w=self.y_ref-self.y_ref_last+self.e
        self.T_in = np.array((0.0, 0.0))
        self.X0 = copy.deepcopy(x_k0[0:2])
        self.delta[0,:]=self.X0-self.x_batch[0,:]
        self.x_batch[0, :]=copy.deepcopy(self.X0)
        # compute the initial output error
        tem=self.C_t[0]@self.x_k0[0:2]
        self.e[0,:]=self.y_ref[0,:]-tem
        # construct the ILI data
        ILI_data=[self.delta[0,:],self.X0,tem]
        return np.hstack((self.delta[0,:],self.w[1:,:].flatten())),tem,self.e[0,:], ILI_data # control state vector, output, error
    def step(self, input_signal):
        ILI_r=copy.deepcopy(input_signal)
        self.T_in[0] = self.T_in[1]
        self.T_in[1] = self.T_in[1] + self.T
        # the current time step
        t = int(self.T_in[1])
        self.control_signal[t-1]=self.control_signal[t-1]+input_signal
        input_signal = np.repeat(self.control_signal[t-1], 2, axis=0)
        t_step, y_step, x_step = control.input_output_response(self.sys, self.T_in, input_signal, X0=self.X0,
                                                               params={"A_t": self.A_t[int(t - 1)],
                                                                       "B_t": self.B_t[int(t - 1)],
                                                                       "C_t": self.C_t[int(t)],
                                                                       "omega_t": self.omega_t[int(t - 1)]},return_x=True)

        self.X0 = x_step[:, 1]
        self.delta[t, :] = self.X0 - self.x_batch[t, :]
        self.x_batch[t, :] = copy.deepcopy(self.X0)
        self.e[t,:]=self.y_ref[t,:]-y_step[1]
        output_infor=y_step[1:]
        control_infor=self.control_signal[t-1,0:]
        error_infor=self.e[t,:]
        ILI_data = [self.delta[t, :],ILI_r, self.X0, output_infor]
        return np.hstack((self.delta[t,:],self.w[t+1:,:].flatten())),output_infor,control_infor,error_infor, ILI_data  # control state vector, output, error




    def reset_Pi_robust(self):
        self.T_in = np.array((0.0, 0.0))
        self.X0 = copy.deepcopy(self.x_k0[0:2])
        self.delta[0,:]=self.X0-self.x_batch[0,:]
        self.x_batch[0, :]=copy.deepcopy(self.X0)
        return np.hstack((self.delta[0,:],self.e[1:,:].flatten())),self.x_k0[2]


    def step_Pi_robust(self, input_signal):
        self.T_in[0] = self.T_in[1]
        self.T_in[1] = self.T_in[1] + self.T
        # the current time step
        t = int(self.T_in[1])
        self.control_signal[t-1]=input_signal[0,0]
        input_signal = np.repeat(self.control_signal[t-1], 2, axis=0)
        t_step, y_step, x_step = control.input_output_response(self.sys, self.T_in, input_signal, X0=self.X0,
                                                               params={"A_t": self.A_t[int(t - 1)],
                                                                       "B_t": self.B_t[int(t - 1)],
                                                                       "C_t": self.C_t[int(t)],
                                                                       "omega_t": self.omega_t[int(t - 1)]},return_x=True)
        self.X0 = x_step[:, 1]
        self.delta[t, :] = self.X0 - self.x_batch[t, :]
        self.x_batch[t, :] = copy.deepcopy(self.X0)
        self.e[t,:]=self.y_ref[t,:]-y_step[1]
        output_infor=y_step[1]
        control_infor=self.control_signal[t-1,0]
        return np.hstack((self.delta[t,:],self.e[t+1:,:].flatten())),output_infor,control_infor
    def close(self):
        pass



    def save_initial_data(self):


        df_x_batch = pd.DataFrame(self.x_batch)
        #pdb.set_trace()
        df_x_batch.to_csv('./Data/x_batch.csv')

        df_delta = pd.DataFrame(self.delta)
        #pdb.set_trace()
        df_delta.to_csv('./Data/delta.csv')

        df_e = pd.DataFrame(self.e)
        #pdb.set_trace()
        df_e.to_csv('./Data/e.csv')

        df_control_signal = pd.DataFrame(self.control_signal)
        #pdb.set_trace()
        df_control_signal.to_csv('./Data/control_signal.csv')


    def load_initial_data(self):

        df_x_batch = pd.read_csv('./Data/x_batch.csv', index_col=0)
        self.x_batch=df_x_batch.to_numpy()

        df_delta = pd.read_csv('./Data/delta.csv', index_col=0)
        self.delta=df_delta.to_numpy()


        df_e = pd.read_csv('./Data/e.csv', index_col=0)
        self.e=df_e.to_numpy()


        df_control_signal = pd.read_csv('./Data/control_signal.csv', index_col=0)
        self.control_signal=df_control_signal.to_numpy()






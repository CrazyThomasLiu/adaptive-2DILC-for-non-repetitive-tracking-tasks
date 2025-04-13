import pdb
import typing
import numpy as np
import copy
import pandas as pd
import csv

'Model-based Non-repetitive Trajectory ILC'
class MBNTILC:

    def __init__(self, batch_length: int, A_t, B_t, C_t, Q_t, R_t):
        self.batch_length=batch_length
        # the state space
        self.A_t = A_t
        self.B_t = B_t
        self.C_t = C_t
        # the dimensions of the state space
        self.n=self.A_t[0].shape[0] # the dimensions of the state variable
        self.m=self.B_t[0].shape[1]   # the dimensions of the input variable
        self.q=self.C_t[0].shape[0]  # the dimensions of the output variable
        # the cost function
        self.Q_t = Q_t
        self.R_t = R_t
        # save the all K_t and P_t
        self.K=[]
        self.P=[]

    def computation(self,A_hat_t,B_hat_t,C_hat_t):
        self.A_t=A_hat_t
        self.B_t = B_hat_t
        self.C_t = C_hat_t
        self.K=[]
        self.P=[]

        # initial conditions t=T-1
        Q_bar_t = copy.deepcopy(self.Q_t[self.batch_length-1])  # Q_bar_T-1=Q_T
        A_bar_t=np.block([[-self.C_t[self.batch_length]@self.A_t[self.batch_length-1],np.eye(self.q)]]) #A_bar_T-1=[-C_{T}A_{T-1},I]
        B_bar_t=-self.C_t[self.batch_length]@self.B_t[self.batch_length-1]  #B_bar_T-1=-C_{T}B_{T-1}


        for time in range(self.batch_length-1, -1, -1):  # [T-1,T-2,...,1,0,-1)
            #K_t
            tem=self.R_t[time]+B_bar_t.T@Q_bar_t@B_bar_t
            tem=np.linalg.inv(tem)
            K_t=tem@B_bar_t.T@Q_bar_t@A_bar_t
            #P_t
            P_t=A_bar_t.T@(Q_bar_t-Q_bar_t@B_bar_t@tem@B_bar_t.T@Q_bar_t)@A_bar_t

            self.K.append(copy.deepcopy(K_t))
            self.P.append(copy.deepcopy(P_t))

            if time!=0:
                # A_bar_t
                A_bar_t=np.block([[self.A_t[time-1],np.zeros((self.n,self.q)),np.zeros((self.n,(self.batch_length-time)*self.q))],
                                  [-self.C_t[time]@self.A_t[time-1],np.eye(self.q),np.zeros((self.q,(self.batch_length-time)*self.q))],
                                  [np.zeros(((self.batch_length-time)*self.q,self.n)),np.zeros(((self.batch_length-time)*self.q,self.q)),np.eye((self.batch_length-time)*self.q)]
                                  ])
                #B_bar_t
                B_bar_t=np.block([[self.B_t[time-1]],
                                 [-self.C_t[time]@self.B_t[time-1]],
                                 [np.zeros(((self.batch_length-time)*self.q,self.m))]])
                # Q_bar_t

                Q_bar_t=np.block([[P_t[0:self.n,0:self.n],np.zeros((self.n,self.q)),P_t[0:self.n,self.n:]],
                              [np.zeros((self.q,self.n)),self.Q_t[time-1],np.zeros((self.q,(self.batch_length-time)*self.q))],
                              [P_t[self.n:,0:self.n],np.zeros(((self.batch_length-time)*self.q,self.q)),P_t[self.n:,self.n:]]])

        self.K.reverse()
        self.P.reverse()



    def save_K(self):

        tem=self.K[0]
        for time in range(1,len(self.K)):
            tem=np.block([[tem,self.K[time]]])
        df = pd.DataFrame(tem)
        df.to_csv('Data/mbilc_K.csv')

    def save_P(self):
        self.P_t_length=0  # the sum of the all time P_t
        for time in range(self.batch_length):
            self.P_t_length+=self.n+self.q*(time+1)

        P_t_tem=np.zeros([self.P[0].shape[0],self.P_t_length])
        tem1=0
        tem2=0
        for time in range(0,self.batch_length):
            tem1=copy.deepcopy(tem2)
            tem2+=self.P[time].shape[0]
            P_t_tem[0:self.P[time].shape[0],tem1:tem2]=copy.deepcopy(self.P[time])
        df = pd.DataFrame(P_t_tem)
        df.to_csv('Data/mbilc_P.csv')


    def load_P(self):
        df_P = pd.read_csv('Data/mbilc_P.csv', index_col=0)
        tem_P=df_P.to_numpy()
        P_from_csv=[]
        tem_1=0
        tem_2=0
        for time in range(self.batch_length):
            tem_1=copy.deepcopy(tem_2)
            tem_2=tem_2+self.n+(self.batch_length-time)*self.q
            tem_3=self.n+(self.batch_length-time)*self.q
            P_from_csv.append(np.matrix(tem_P[0:tem_3,tem_1:tem_2]))
        self.P=P_from_csv


    def load_K(self):
        df_K = pd.read_csv('Data/mbilc_K.csv', index_col=0)
        tem_K=df_K.to_numpy()
        K_from_csv=[]
        tem_1=0
        tem_2=0
        for time in range(self.batch_length):
            tem_1=copy.deepcopy(tem_2)
            tem_2=tem_2+self.n+(self.batch_length-time)*self.q
            K_from_csv.append(np.matrix(tem_K[:,tem_1:tem_2]))
        self.K=K_from_csv

import copy
import pdb
import control
from algorithm.mb_nontra_ILC import MBNTILC
import numpy as np
from env.time_varying_batch_sys import BatchSys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import math
import pandas as pd
import csv
# save the figure or not

save_figure=False
save_csv=True
batch_length = 100
batch_num=20

"1. the state space of the time-varying batch systems"
n=2 # state
m=1 # input
q=1 # output
# cost function
Q = np.array([[10.]])
R = np.array([[1.]])

# the time-varying cost function weighting
Q_t=np.zeros((batch_length,q,q))
R_t=np.zeros((batch_length,m,m))

for time in range(batch_length):
    Q_t[time]=copy.deepcopy(Q)
    R_t[time] = copy.deepcopy(R)

'1.1 the time-invariant parameters'
A= np.array([[1.607, 1.0], [-0.6086, 0.0]])
B= np.array([[1.239], [-0.9282]])
C= np.array([[1.0,0.0]])
delta_C_t=np.array([[0.01,-0.01]])
'1.2 the time-varying parameters'
#
A_t = np.zeros((batch_length,n,n))
B_t = np.zeros((batch_length,n,m))
C_t = np.zeros((batch_length+1,q,n))
for time in range(batch_length):
    A_t[time]=A*(0.5+0.2*np.exp(time/200))
    B_t[time]= B*(1+0.2*np.exp(time/100))
    C_t[time] = C+ delta_C_t * np.sin(time)
'1.3 add a additional parameter for the t+1'
C_t[batch_length] = C+ delta_C_t * np.sin(batch_length)

'1.4 set the time-varying disturbance'
omega_t = np.ones((batch_length, n))
for time in range(batch_length):
    omega_t[time,0]=omega_t[time,0]*5*np.sin(0.1*time)
    omega_t[time, 1] = omega_t[time, 1] *5* np.cos(0.1*time)

"2. get model-based optimal ILC for the non-repetitive trajectory"
mbilc = MBNTILC(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t, Q_t=Q_t, R_t=R_t)
mbilc.computation(A_hat_t=A_t, B_hat_t=B_t, C_hat_t=C_t)
"3. set the simulated batch system"
'3.1 the trajectory'

y_ref_standard=np.ones((batch_length+1, q))
y_ref_standard[0:51] = 120. * y_ref_standard[0:51]
y_ref_standard[51:] = 150. * y_ref_standard[51:]
'3.2 the boundary conditions and sample time'
x_k0 = np.array((10., 20., 0.))
sample_time = 1


'3.3 set the time-varying batch system'
def state_update(t, x, u, params):
    # get the parameter from the params
    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # Get the current time t state space
    A_t_env=params.get('A_t')
    B_t_env = params.get('B_t')
    omega_t_env=params.get('omega_t')
    # Compute the discrete updates
    dz1 = A_t_env[0,0]* z1 + A_t_env[0,1]* z2 +B_t_env[0,0]*u+omega_t_env[0]
    dz2 = A_t_env[1,0]* z1 + A_t_env[1,1]* z2 +B_t_env[1,0]*u+omega_t_env[1]
    return [dz1, dz2]


def output_update(t, x, u, params):
    # Parameter setup
    # Compute the discrete updates
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    C_t_env = params.get('C_t')
    y1 = C_t_env[0,0]* z1 + C_t_env[0,1]* z2

    return [y1]

batch_sys = control.NonlinearIOSystem(
    state_update, output_update, inputs=('u'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=1, name='Injection_Molding')

controlled_system = BatchSys(batch_length=batch_length, sample_time=sample_time, sys=batch_sys, x_k0=x_k0, A_t=A_t,B_t=B_t,C_t=C_t,y_ref=y_ref_standard,omega_t=omega_t)

"4. Simulations"

y_batchdata=np.zeros((batch_num,batch_length+1,q))
u_batchdata=np.zeros((batch_num,batch_length,m))
e_batchdata=np.zeros((batch_num,batch_length+1,q))
for batch in range(batch_num):
    # define the non-repetitive trajectory
    y_ref = y_ref_standard*(1+0.1*np.sin(batch))
    x_k0_random = x_k0
    state, output_info, error_info, ILI_info = controlled_system.reset_randomly(y_ref=y_ref, x_k0=x_k0_random)
    y_batchdata[batch,0,:]=copy.deepcopy(output_info)
    e_batchdata[batch, 0, :] = copy.deepcopy(error_info)
    for time in range(batch_length):
        control_signal = -mbilc.K[time]@state
        state,output_info,control_infor,error_info, info=controlled_system.step(control_signal)
        y_batchdata[batch, time+1, :] = copy.deepcopy(output_info)
        e_batchdata[batch, time + 1, :] = copy.deepcopy(error_info)
        u_batchdata[batch, time, :] = copy.deepcopy(control_infor)

'5 the RMSE for the trajectory'

RMSE=np.zeros(batch_num)
for batch in range(batch_num):
    for time in range(batch_length):
        RMSE[batch] += (abs(e_batchdata[batch, time + 1, 0])) ** 2
    RMSE[batch] = math.sqrt(RMSE[batch] / batch_length)

if save_csv == True:
    df_RMSE = pd.DataFrame(RMSE)
    df_RMSE.to_csv('RMSE_mbilc_paper.csv')

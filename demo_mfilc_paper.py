import copy
import pdb
import random
import control
import os
import sys
from algorithm.mb_nontra_ILC import MBNTILC
import numpy as np
from env.time_varying_batch_sys import BatchSys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import math
from algorithm.iterative_learning_identification import ILI
import pandas as pd
import csv
# save the figure or not

save_figure=True
save_csv=True
batch_length = 100
batch_num=20

# set the fixed random seed

np.random.seed(1)

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




"2. set the related parameters of the estimated model"
'2.1 the weighting matrix of the ILI'
# cost function
Q_bar_1 = np.array([[10.]])
R_bar_1 = np.array([[2.,0.,0.],
                  [0.,2.,0.],
                  [0.,0.,2.]])

# the time-varying cost function weighting
Q_bar_t_1=np.zeros((batch_length,1,1))
R_bar_t_1=np.zeros((batch_length,n+m,n+m))

for time in range(batch_length):
    Q_bar_t_1[time]=copy.deepcopy(Q_bar_1)
    R_bar_t_1[time] = copy.deepcopy(R_bar_1)

# cost function
Q_bar_2 = np.array([[10.]])
R_bar_2 = np.array([[2.,0.],
                  [0.,2.]])

# the time-varying cost function weighting
Q_bar_t_2=np.zeros((batch_length,1,1))
R_bar_t_2=np.zeros((batch_length,n,n))

for time in range(batch_length):
    Q_bar_t_2[time]=copy.deepcopy(Q_bar_2)
    R_bar_t_2[time] = copy.deepcopy(R_bar_2)

'2.2 the initial system model'
A_hat_t = np.zeros((batch_length,n,n))
B_hat_t = np.zeros((batch_length,n,m))
C_hat_t = np.zeros((batch_length+1,q,n))

for time in range(batch_length):
    A_hat_t[time]=A*0.65
    B_hat_t[time] = B * 0.8
    C_hat_t[time] = C+ delta_C_t

'2.3 add a additional parameter for the t+1'
C_hat_t[batch_length] = C+ delta_C_t

"3. get model-based optimal ILC for the non-repetitive trajectory"
mbilc = MBNTILC(batch_length=batch_length, A_t=A_hat_t, B_t=B_hat_t, C_t=C_hat_t, Q_t=Q_t, R_t=R_t)
mbilc.computation(A_hat_t=A_hat_t,B_hat_t=B_hat_t,C_hat_t=C_hat_t)


"4. set the simulated batch system"
'4.1 the trajectory'



y_ref_standard=np.ones((batch_length+1, q))
y_ref_standard[0:51] = 120. * y_ref_standard[0:51]
y_ref_standard[51:] = 150. * y_ref_standard[51:]

'4.2 the boundary conditions and sample time'
x_k0 = np.array((10., 20., 0.)) # the 2 state and 1 output
sample_time = 1


'4.3 set the time-varying batch system'
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

"5. Simulations"

y_batchdata=np.zeros((batch_num,batch_length+1,q))
u_batchdata=np.zeros((batch_num,batch_length,m))
e_batchdata=np.zeros((batch_num,batch_length+1,q))
#pdb.set_trace()
# define the data for ILI
ILI_delta=np.ones((batch_length+1,n))
ILI_r=np.ones((batch_length,m))
ILI_x=np.ones((batch_length+1,n))
ILI_y=np.ones((batch_length+1,q))
sys_model = ILI(batch_length=batch_length, A_hat_t=A_hat_t, B_hat_t=B_hat_t, C_hat_t=C_hat_t, Q_bar_t_1=Q_bar_t_1, R_bar_t_1=R_bar_t_1,Q_bar_t_2=Q_bar_t_2, R_bar_t_2=R_bar_t_2)
for batch in range(batch_num):
    # define the non-repetitive trajectory
    y_ref = y_ref_standard*(1+0.1*np.sin(batch))


    x_k0_random = x_k0
    state, output_info, error_info, ILI_info = controlled_system.reset_randomly(y_ref=y_ref,x_k0=x_k0_random)
    y_batchdata[batch,0,:]=copy.deepcopy(output_info)
    e_batchdata[batch, 0, :] = copy.deepcopy(error_info)
    ILI_delta[0]=ILI_info[0]
    ILI_x[0] = ILI_info[1]
    ILI_y[0] = ILI_info[2]
    for time in range(batch_length):
        control_signal = -mbilc.K[time]@state
        state,output_info,control_infor,error_info,ILI_info=controlled_system.step(control_signal)
        y_batchdata[batch, time+1, :] = copy.deepcopy(output_info)
        e_batchdata[batch, time + 1, :] = copy.deepcopy(error_info)
        u_batchdata[batch, time, :] = copy.deepcopy(control_infor)
        ILI_delta[time+1] = ILI_info[0]
        ILI_r[time] = ILI_info[1]
        ILI_x[time+1] = ILI_info[2]
        ILI_y[time+1] = ILI_info[3]

    A_hat_t,B_hat_t,C_hat_t=sys_model.iterative_model(delta=ILI_delta,r=ILI_r,x=ILI_x,y=ILI_y)
    mbilc.computation(A_hat_t=A_hat_t, B_hat_t=B_hat_t, C_hat_t=C_hat_t)

'6.1 the output trajectory for the 1D batch system'
# calculate the real trajectory for the 1D batch system
plt.rcParams['pdf.fonttype'] = 42
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 14,
         }
fig=plt.figure()
ax1=plt.axes(projection="3d")
ax1.invert_xaxis()
x_major_locator=MultipleLocator(2)
ax1.xaxis.set_major_locator(x_major_locator)

t=range(batch_length+1)     # initial state 1 + time length 100
batch_before=np.ones(batch_length+1,dtype=int)
batch_plot=batch_before*(batch_num)
for batch in range(batch_num):
    if batch%2==0:
        batch_plot=batch_before*(batch+1)
        ax1.plot3D(batch_plot, t, y_batchdata[batch, :, 0], linewidth=1, color='black')
        y_ref = y_ref_standard * (1 + 0.1 * np.sin(batch))
        ax1.plot3D(batch_plot, t, y_ref[:,0], linewidth=1.2, color='green',linestyle = 'dashdot')

xlable = 'Batch:$k$'
ylable = 'Time:$t$'
zlable = 'Output:$y_{k,t}$'
ax1.set_xlabel(xlable,font2)
ax1.set_ylabel(ylable,font2)
ax1.set_zlabel(zlable,font2)
ax1.legend(['Output Response','Reference Trajectory'],bbox_to_anchor=(1.1, 1.1),loc=1)
ax1.view_init(29, -47)
plt.tick_params(labelsize=11)
if save_figure==True:
    plt.savefig('mfilc_response_paper.pdf')


'5.2 the control signal for the 1D batch system'
fi2=plt.figure()
ax2=plt.axes(projection="3d")
ax2.invert_xaxis()
x_major_locator=MultipleLocator(2)
ax2.xaxis.set_major_locator(x_major_locator)

t=range(batch_length)     # initial state 1 + time length 100
batch_before=np.ones(batch_length,dtype=int)
batch_plot=batch_before*(batch_num)
for batch in range(batch_num):
    if batch % 2 == 0:
        batch_plot=batch_before*(batch+1)
        ax2.plot3D(batch_plot, t, u_batchdata[batch,:,0], linewidth=0.8, color='black')
xlable = 'Batch:$k$'
ylable = 'Time:$t$'
zlable = 'Control Signal:$u_{k,t}$'
ax2.set_xlabel(xlable,font2)
ax2.set_ylabel(ylable,font2)
ax2.set_zlabel(zlable,font2)

ax2.view_init(39, -47)
if save_figure==True:
    plt.savefig('mfilc_control_signal_paper.pdf')







'5.3 the RMSE for the trajectory'

RMSE=np.zeros(batch_num)
for batch in range(batch_num):
    for time in range(batch_length):
        RMSE[batch] += (abs(e_batchdata[batch, time + 1, 0])) ** 2
    RMSE[batch] = math.sqrt(RMSE[batch] / batch_length)
if save_csv == True:
    df_RMSE = pd.DataFrame(RMSE)
    df_RMSE.to_csv('RMSE_mfilc_paper.csv')

plt.show()
pdb.set_trace()
a=2
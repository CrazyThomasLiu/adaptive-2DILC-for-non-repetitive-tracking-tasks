import copy
import pdb
import random

import control
import os
import sys
#pdb.set_trace()
#config_path=os.path.split(os.path.abspath(__file__))[0]
#config_path=config_path.rsplit('/',1)[0]
#sys.path.append(config_path)
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
batch_num=50

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

#pdb.set_trace()
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
    #A_t.append(copy.deepcopy(A))
    A_t[time]=A*(0.5+0.2*np.exp(time/200))
    #A_t[time][0, 0] = A_t[time][0, 0] * np.exp(time / 100)
    #B_t.append(copy.deepcopy(B))
    B_t[time]= B*(1+0.1*np.exp(time/50))
    #B_t[time][0, 0] = B_t[time][0, 0] * np.exp(time / 100)
    #C_t.append(copy.deepcopy(C))
    C_t[time] = C+ delta_C_t * np.sin(time)
'1.3 add a additional parameter for the t+1'
C_t[batch_length] = C+ delta_C_t * np.sin(batch_length)

'1.4 set the time-varying disturbance'
omega_t = np.ones((batch_length, n))
for time in range(batch_length):
    omega_t[time,0]=omega_t[time,0]*20*np.sin(time)
    omega_t[time, 1] = omega_t[time, 1] *20* np.cos(time)

"2. get model-based optimal ILC for the non-repetitive trajectory"
mbilc = MBNTILC(batch_length=batch_length, A_t=A_t, B_t=B_t, C_t=C_t, Q_t=Q_t, R_t=R_t)
#pdb.set_trace()
#mbilc.load_K()
#mbilc.load_P()
mbilc.computation(A_hat_t=A_t, B_hat_t=B_t, C_hat_t=C_t)
#mbilc.save_K()
#mbilc.save_P()

"2. set the simulated batch system"
'2.1 the trajectory'


#y_ref_standard=200.*np.ones((batch_length+1, q))
y_ref_standard=np.ones((batch_length+1, q))
y_ref_standard[0:51] = 100. * y_ref_standard[0:51]
y_ref_standard[51:] = 150. * y_ref_standard[51:]
#y_ref_standard[0,0]=0.
#y_ref_standard[1:26] = 200. * y_ref_standard[1:26]
#y_ref_standard[26:] = 250. * y_ref_standard[26:]
#pdb.set_trace()
#y_ref = np.ones((batch_length+1, 1))
#y_ref[0,0]=0.
#y_ref[1:26] = 200. * y_ref[1:26]
#y_ref[26:] = 250. * y_ref[26:]
#pdb.set_trace()
'2.2 the boundary conditions and sample time'
#x_k0 = np.array((0., 0., 0.)) # the 2 state and 1 output
x_k0 = np.array((10., 20., 0.))
sample_time = 1


'2.3 set the time-varying batch system'
def state_update(t, x, u, params):
    # get the parameter from the params
    # Map the states into local variable names
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # Get the current time t state space
    A_t_env=params.get('A_t')
    B_t_env = params.get('B_t')
    omega_t_env=params.get('omega_t')
    #pdb.set_trace()
    # Compute the discrete updates
    dz1 = A_t_env[0,0]* z1 + A_t_env[0,1]* z2 +B_t_env[0,0]*u+omega_t_env[0]
    dz2 = A_t_env[1,0]* z1 + A_t_env[1,1]* z2 +B_t_env[1,0]*u+omega_t_env[1]
    # pdb.set_trace()
    return [dz1, dz2]


def output_update(t, x, u, params):
    # Parameter setup
    # pdb.set_trace()
    # Compute the discrete updates
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    C_t_env = params.get('C_t')
    #pdb.set_trace()
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
#pdb.set_trace()
for batch in range(batch_num):
    # define the non-repetitive trajectory
    y_ref = y_ref_standard*(1+0.05*np.sin(batch))
    x_k0_random = x_k0
    #print(y_ref)
    #pdb.set_trace()
    state, output_info, error_info, ILI_info = controlled_system.reset_randomly(y_ref=y_ref, x_k0=x_k0_random)
    #state, output_info,error_info, info = controlled_system.reset(y_ref=y_ref)
    y_batchdata[batch,0,:]=copy.deepcopy(output_info)
    e_batchdata[batch, 0, :] = copy.deepcopy(error_info)
    #pdb.set_trace()
    for time in range(batch_length):
        control_signal = -mbilc.K[time]@state
        #pdb.set_trace()
        state,output_info,control_infor,error_info, info=controlled_system.step(control_signal)
        y_batchdata[batch, time+1, :] = copy.deepcopy(output_info)
        e_batchdata[batch, time + 1, :] = copy.deepcopy(error_info)
        #pdb.set_trace()
        u_batchdata[batch, time, :] = copy.deepcopy(control_infor)
        #pdb.set_trace()
    #pdb.set_trace()
#pdb.set_trace()
'5.1 the output trajectory for the 1D batch system'
# calculate the real trajectory for the 1D batch system
#sum_batch_state_y=sum_batch_state_y+150.
font2 = {'family': 'Arial',
         'weight': 'bold',
         'size': 14,
         }
fig=plt.figure()
ax1=plt.axes(projection="3d")
ax1.invert_xaxis()
x_major_locator=MultipleLocator(4)
ax1.xaxis.set_major_locator(x_major_locator)

#ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(batch_length+1)     # initial state 1 + time length 100
batch_before=np.ones(batch_length+1,dtype=int)
batch_plot=batch_before*(batch_num)
#pdb.set_trace()
#ax1.plot3D(batch_plot,t, y_ref[:,0],linewidth=2,color='green',linestyle = 'dashdot')
for batch in range(batch_num):
    batch_plot=batch_before*(batch+1)
    ax1.plot3D(batch_plot, t, y_batchdata[batch,:,0], linewidth=1, color='black')
    #if (batch%2)==0:
    #    ax1.plot3D(batch_plot,t, sum_batch_state_y[batch],linewidth=1,color='black')
#ax1.plot3D(batch_plot, t, y_batchdata[0,:,0], linewidth=1, color='black')
xlable = 'Batch:$k$'
ylable = 'Time:$t$'
zlable = 'Output:$y$'
ax1.set_xlabel(xlable,font2)
ax1.set_ylabel(ylable,font2)
ax1.set_zlabel(zlable,font2)
ax1.legend(['Reference Trajectory','Output Response'],bbox_to_anchor=(1.1, 1.1),loc=1)
ax1.view_init(24, -27)
#ax1.set_title('10-iteration $x_{1}$')
if save_figure==True:
    plt.savefig('mfoilc_response_PILC.pdf')
    #plt.savefig('mfoilc_response_PILC.jpg',dpi=700)


'5.2 the output trajectory for the 1D batch system'
fi2=plt.figure()
ax2=plt.axes(projection="3d")
ax2.invert_xaxis()
x_major_locator=MultipleLocator(4)
ax2.xaxis.set_major_locator(x_major_locator)

#ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(batch_length)     # initial state 1 + time length 100
batch_before=np.ones(batch_length,dtype=int)
batch_plot=batch_before*(batch_num)
#pdb.set_trace()
#ax1.plot3D(batch_plot,t, y_ref[:,0],linewidth=2,color='green',linestyle = 'dashdot')
for batch in range(batch_num):
    batch_plot=batch_before*(batch+1)
    ax2.plot3D(batch_plot, t, u_batchdata[batch,:,0], linewidth=1, color='black')
    #if (batch%2)==0:
    #    ax1.plot3D(batch_plot,t, sum_batch_state_y[batch],linewidth=1,color='black')
#ax1.plot3D(batch_plot, t, y_batchdata[0,:,0], linewidth=1, color='black')
xlable = 'Batch:$k$'
ylable = 'Time:$t$'
zlable = 'Control Signal:$u$'
ax2.set_xlabel(xlable,font2)
ax2.set_ylabel(ylable,font2)
ax2.set_zlabel(zlable,font2)
ax2.legend(['Reference Trajectory','Output Response'],bbox_to_anchor=(1.1, 1.1),loc=1)
ax2.view_init(24, -27)
#ax1.set_title('10-iteration $x_{1}$')
if save_figure==True:
    plt.savefig('mfoilc_response_PILC.pdf')
    #plt.savefig('mfoilc_response_PILC.jpg',dpi=700)

'5.3 the tracking error'
fi3=plt.figure()
ax3=plt.axes(projection="3d")
ax3.invert_xaxis()
x_major_locator=MultipleLocator(4)
ax3.xaxis.set_major_locator(x_major_locator)

#ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
t=range(batch_length)     # initial state 1 + time length 100
batch_before=np.ones(batch_length,dtype=int)
batch_plot=batch_before*(batch_num)
#pdb.set_trace()
#ax1.plot3D(batch_plot,t, y_ref[:,0],linewidth=2,color='green',linestyle = 'dashdot')
for batch in range(batch_num):
    batch_plot=batch_before*(batch+1)
    ax3.plot3D(batch_plot, t, e_batchdata[batch,1:,0], linewidth=1, color='black')
    #if (batch%2)==0:
    #    ax1.plot3D(batch_plot,t, sum_batch_state_y[batch],linewidth=1,color='black')
#ax1.plot3D(batch_plot, t, y_batchdata[0,:,0], linewidth=1, color='black')
xlable = 'Batch:$k$'
ylable = 'Time:$t$'
zlable = 'Tracking error:$e$'
ax3.set_xlabel(xlable,font2)
ax3.set_ylabel(ylable,font2)
ax3.set_zlabel(zlable,font2)
ax3.legend(['Reference Trajectory','Output Response'],bbox_to_anchor=(1.1, 1.1),loc=1)
ax3.view_init(24, -27)
#ax1.set_title('10-iteration $x_{1}$')
if save_figure==True:
    plt.savefig('mfoilc_response_PILC.pdf')
    #plt.savefig('mfoilc_response_PILC.jpg',dpi=700)


'5.4 the RMSE for the trajectory'

RMSE=np.zeros(batch_num)
for batch in range(batch_num):
    #y_batch = sum_batch_state_y[batch]
    for time in range(batch_length):
        #pdb.set_trace()
        RMSE[batch] += (abs(e_batchdata[batch, time + 1, 0])) ** 2
    RMSE[batch] = math.sqrt(RMSE[batch] / batch_length)

#pdb.set_trace()

batch_time=range(1,batch_num+1)
fig4=plt.figure(figsize=(7,5.5))
x_major_locator=MultipleLocator(int(batch_num/1))
ax4=plt.gca()
ax4.xaxis.set_major_locator(x_major_locator)
plt.plot(batch_time,RMSE,linewidth=1.5,color='tab:blue',linestyle = 'dashdot')
#plt.plot(batch_time,y_2dilc_rl_show,linewidth=1.5,color='tab:orange',linestyle='solid')
plt.grid()

xlable = 'Batch:$\mathit{k} $'
ylable = 'RMSE'

plt.xlabel(xlable,font2 )
plt.ylabel(ylable,font2 )
#ax3.set_title('10-iteration')
#plt.legend(['2D Iterative Learning Control Scheme','2D ILC-RL Control Scheme'])
if save_figure==True:
    plt.savefig('mfoilc_RMSE_PILC.pdf')
    #plt.savefig('mfoilc_RMSE_PILC.jpg',dpi=700)

if save_csv == True:
    df_RMSE = pd.DataFrame(RMSE)
    df_RMSE.to_csv('RMSE_mbilc.csv')
plt.show()
pdb.set_trace()
a=2
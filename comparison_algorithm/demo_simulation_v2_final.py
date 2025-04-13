import copy
import pdb
import control
import os
import sys
config_path=os.path.split(os.path.abspath(__file__))[0]
config_path=config_path.rsplit('/',1)[0]
sys.path.append(config_path)
from env.time_varying_batch_sys import BatchSys
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
# save the figure or not

save_figure=False
save_csv=False
batch_length = 100
batch_num=50
"1. the state space of the injection molding"
'1.1 the time-invariant parameters'
A= np.matrix([[1.607, 1.0], [-0.6086, 0.0]])
B= np.matrix([[1.239], [-0.9282]])
C= np.matrix([[1.0,0.0]])
delta_C_t=np.matrix([[0.01,-0.01]])
n=2 # state
m=1 # input
q=1 # output
'1.2 the time-varying parameters'
#A_t = []
#B_t = []
#C_t=  []
A_t = np.zeros((batch_length,n,n))
B_t = np.zeros((batch_length,n,m))
C_t = np.zeros((batch_length+1,q,n))
for time in range(batch_length):
    #A_t.append(copy.deepcopy(A))
    A_t[time]=A*(0.5+0.2*np.exp(time/200))
    #A_t[time][0, 0] = A_t[time][0, 0] * np.exp(time / 100)
    #B_t.append(copy.deepcopy(B))
    B_t[time]= B*(1+0.2*np.exp(time/100))
    #B_t[time][0, 0] = B_t[time][0, 0] * np.exp(time / 100)
    #C_t.append(copy.deepcopy(C))
    C_t[time] = C+ delta_C_t * np.sin(time)
'1.3 add a additional parameter for the t+1'
C_t[batch_length] = C+ delta_C_t * np.sin(batch_length)
'1.4 the reference trajectory'
y_ref_standard=np.ones((batch_length+1, q))
y_ref_standard[0:51] = 120. * y_ref_standard[0:51]
y_ref_standard[51:] = 150. * y_ref_standard[51:]

"2. set the PI controller"

#pdb.set_trace()
# robust control parameters

#Kp [0.42925293]
#Ki 0.07341834378229248

K_p=0.42925293
K_i= 0.07341834378229248

# robust control parameters
#L1 [[-0.14558297]]
#L2 [[-6.38077942e-14]]
#L3 [[1.0095119]]



L1=-0.14558297
L2=-6.38077942e-14
L3=1.0095119


"3. set the simulated env"
'3.1 the initial state'
x_k0 = np.array((10., 20., 0.))# the 2 state and 1 output
sample_time = 1

'3.2 set the time-varying disturbance'
omega_t = np.ones((batch_length, n))
for time in range(batch_length):
    omega_t[time,0]=omega_t[time,0]*5*np.sin(0.1*time)
    omega_t[time, 1] = omega_t[time, 1] *5* np.cos(0.1*time)
#pdb.set_trace()
"3.2 set the batch system"
def state_update(t, x, u, params):
    # get the parameter from the params
    # pdb.set_trace()
    # Map the states into local variable names
    # the state x_{k+1,t}
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    # Get the current time t state space
    A_t_env=params.get('A_t')
    B_t_env = params.get('B_t')
    omega_t_env=params.get('omega_t')
    # Compute the discrete updates
    dz1 = A_t_env[0,0]* z1 + A_t_env[0,1]* z2 +B_t_env[0,0]*u+omega_t_env[0]
    dz2 = A_t_env[1,0]* z1 + A_t_env[1,1]* z2 +B_t_env[1,0]*u+omega_t_env[1]
    # pdb.set_trace()
    return [dz1, dz2]


def ouput_update(t, x, u, params):
    # Parameter setup
    # Compute the discrete updates
    z1 = np.array([x[0]])
    z2 = np.array([x[1]])
    C_t_env = params.get('C_t')
    y1 = C_t_env[0,0]* z1 + C_t_env[0,1]* z2

    return [y1]

batch_sys = control.NonlinearIOSystem(
    state_update, ouput_update, inputs=('u'), outputs=('y1'),
    states=('dz1', 'dz2'), dt=1, name='Injection_Molding')

controlled_system = BatchSys(batch_length=batch_length, sample_time=sample_time, sys=batch_sys,x_k0=x_k0,A_t=A_t, B_t=B_t, C_t=C_t,y_ref=y_ref_standard,omega_t=omega_t)


"4. simulations"
e_k_list = np.zeros(batch_length + 1)
e_k_1_list = np.zeros(batch_length + 1)

# set the e_k_1_list
e_k_1_list[:] = y_ref_standard[:, 0]
# define the sum of the e_{s} not the e
# the initial e_s_sum=0
'for e_s_sum 0=-1  1=0'
e_s_k_sum_list = np.zeros(batch_length + 1)
e_s_k_1_sum_list = np.zeros(batch_length + 1)
# define the y_s
# the initial y_s=0
ys_k_list = np.zeros(batch_length)
ys_k_1_list = copy.deepcopy(0.85 * y_ref_standard[:,0])
RMSE = np.zeros(batch_num)
state_u=[]
y_out_list=[]
y_out=[]
u_out_list=[]
u_out=[]
for batch in range(batch_num):

    y_ref = y_ref_standard * (1 + 0.1 * np.sin(batch))
    # reset the sum of the current error
    e_s_sum = 0
    x_tem, y_current = controlled_system.reset_Pi_robust()
    y_out=[]
    u_out = []
    y_out.append(copy.deepcopy(y_current))
    e_current = y_ref[0,0] - y_current
    e_k_list[0] = copy.deepcopy(e_current)
    for time in range(batch_length):
        # e_sum_current
        # delta_e_s
        delta_e_s = e_s_k_sum_list[time] - e_s_k_1_sum_list[time]
        'y_s'
        y_s = ys_k_1_list[time] + L1 * delta_e_s + L2 * e_current + L3 * e_k_1_list[time + 1]
        #pdb.set_trace()
        ys_k_list[time] = copy.deepcopy(y_s)
        # e_s
        e_s = y_s - y_current
        # e_s_sum
        e_s_sum = e_s_sum + e_s
        'using the last time'
        e_s_k_sum_list[time+1] = copy.deepcopy(e_s_sum)
        # u
        u = K_p * e_s + K_i * e_s_sum
        control_signal=np.matrix([[u]])

        x_tem,y_current,control_info = controlled_system.step_Pi_robust(control_signal)
        y_out.append(copy.deepcopy(y_current))
        u_out.append(copy.deepcopy(control_signal[0, 0]))
        # e
        e_current = y_ref[time+1,0] - y_current
        e_k_list[time + 1] = copy.deepcopy(e_current)
    y_out_list.append(copy.deepcopy(y_out))
    u_out_list.append(copy.deepcopy(u_out))
    e_k_1_list = copy.deepcopy(e_k_list)
    e_s_k_1_sum_list = copy.deepcopy(e_s_k_sum_list)
    ys_k_1_list = copy.deepcopy(ys_k_list)
    # calculation of the RMSE
    tem = 0.0
    for time in range(batch_length):
        tem += (abs(y_out_list[batch][time+1]-y_ref[time+1,0])) ** 2
        RMSE[batch] = math.sqrt(tem / batch_length)

if save_csv == True:
    df_RMSE = pd.DataFrame(RMSE)
    df_RMSE.to_csv('RMSE_PI_Robust_v2_final.csv')

pdb.set_trace()
a=2
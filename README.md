# Adaptive Two-dimensional Iterative Learning Control for Time-varying Batch Processes with Non-repetitive Tracking Tasks

## Catalog
* env: The time-varying batch process.
* algorithm:  The proposed control schemes: the model-based two-dimensional Iterative Learning Control and the model-free adaptive two-dimensional Iterative Learning Control. 
* policy: the training policy saved in this folder.
* comparison_method: The comparison control scheme: PI-based indirect-type ILC from paper 'PI based indirect-type iterative learning control for batch processes with time-varying uncertainties: A 2D FM model based approach' Journal of process control,2019
## Getting Started
* Clone this repo: `git clone https://github.com/CrazyThomasLiu/adaptive-2DILC-for-non-repetitive-tracking-tasks.git`
* Create a python virtual environment and activate. `conda create -n 2dilc python=3.10` and `conda activate 2dilc`
* Install dependenices. `cd adaptive-2DILC-for-non-repetitive-tracking-tasks`, `pip install -r requirement.txt` 


## Simulation in the time-varying injection molding with non-repetitive tracking tasks.
*  the model-based two-dimensional Iterative Learning Control
```
python demo_mboilc.py
```
* the model-free adaptive two-dimensional Iterative Learning Control. 
```
python demo_mfilc.py
```
![image](https://github.com/CrazyThomasLiu/adaptive-2DILC-for-non-repetitive-tracking-tasks/blob/master/mfilc_response_paper.jpg)
![image](https://github.com/CrazyThomasLiu/adaptive-2DILC-for-non-repetitive-tracking-tasks/blob/master/mfilc_control_signal_paper.jpg)



## Run the comparison method PI-based indirect-type ILC
Run the following command to obtain control law.
```
python pi_controller_paper.py 
```

```
python robust_pi_controller_paper.py 
```
Simulate the injection molding process with non-repetitive tracking tasks under the PI-based indirect-type ILC
```
python demo_simulation_paper.py
```

## Test for the control performance 
Compare the control performance between the model-based two-dimensional Iterative Learning Control, the model-free adaptive two-dimensional Iterative Learning Control, and the PI-based indirect ILC.

```
python demo_compare_RMSE.py
```
![image](https://github.com/CrazyThomasLiu/adaptive-2DILC-for-non-repetitive-tracking-tasks/blob/master/RMSE_paper.jpg)





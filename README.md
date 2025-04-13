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
![image](https://github.com/CrazyThomasLiu/2d_linear_oc/blob/master/workspace2/mfoilc_input.jpg)
![image](https://github.com/CrazyThomasLiu/2d_linear_oc/blob/master/workspace2/mfoilc_response.jpg)



## Sample the operation batch data
Run the following command to obtain the operation data by using the model-free P-type ILC.
```
python demo_pilc_sample.py 
```

## Test for the control performance
Compare the control performance between the 2D model-based optimal ILC, the model-free off-policy optimal ILC, and the PI-based indirect-type ILC.

```
python demo_compare_RMSE.py
```
![image](https://github.com/CrazyThomasLiu/2d_linear_oc/blob/master/workspace2/Compare_RMSE.jpg)




# Citation:
The 2D Iterative Learning Control with Deep Reinforcement Learning Compensation for the Non-repetitive Uncertainty Batch Processes was published in the Journal of Process Control.


```
Liu J, Zhou Z, Hong W, et al. Two-dimensional iterative learning control with deep reinforcement learning compensation for the non-repetitive uncertain batch processes[J]. Journal of Process Control, 2023, 131: 103106.
```



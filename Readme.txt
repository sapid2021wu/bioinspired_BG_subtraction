EMD-based Object Detection model  
This is a MatLab implementation of the EMD-based model for moving object detection. 
Details regarding the model can be found in the following paper: 
Zhu’anzhen Zheng, Aike Guo, Zhihua Wu. "Moving Object Detection Based on Bioinspired Background Subtraction". accepted by Bioinspiration & Biomimetics in June 2024.

Code 
The main programs are EMD_LoLP_Jitter.m and EMD_LoLP_aeroplane13.m, which are self explanatory. 
The subfolders '\artificialStimuli' and '\Dataset' store respective visual stimuli.
Each of the two programs needs to call 5 function programs as follows.
* Rect.m
* RK4.m
* F_measure.m
* LP_Units.m
* Lo_Units_Jitter.m  (only needed by EMD_LoLP_Jitter.m)
* Lo_Units_aeroplane13.m (only needed by EMD_LoLP_aeroplane13.m)

How to run
1) Run EMD_LoLP_Jitter.m
* Start MATLAB and navigate to project directory.
* '>> EMD_LoLP_Jitter'.
* Four figures will popup, which reproduce the simulation results in Figure 5 in the paper.

2) Run EMD_LoLP_aeroplane13.m
* Start MATLAB and navigate to project directory.
* '>> EMD_LoLP_aeroplane13'.
* Two figures will popup, which reproduce the simulation results in Figure 7 in the paper (with the stimulus aeroplane13.avi).

Written by Zhu’anzhen Zheng and Zhihua Wu

License
GPL-3.0 

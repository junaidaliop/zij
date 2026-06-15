# PID

Implements PID, an SGD variant cast as a PID controller.

Classical momentum is the integral term of a PID controller acting on the
gradient. PID adds the derivative term, the momentum-smoothed change in the
gradient between consecutive steps, so that the update reacts to both the
accumulated history and the instantaneous trend of the gradient:


$$
\begin{aligned}
     I_t &= \mu I_{t-1} + (1 - \tau) g_t                                 \\
     D_t &= \mu D_{t-1} + (1 - \mu)(g_t - g_{t-1})                        \\
     \theta_t &= \theta_{t-1} - \eta \left( g_t + k_i I_t + k_d D_t
         \right)
\end{aligned}
$$

where $\mu$ is the momentum, $\tau$ the dampening,
$k_i$ the integral gain, and $k_d$ the derivative gain.

Reference: Wangpeng An, Haoqian Wang, Qingyun Sun, Jun Xu, Qionghai Dai,
Lei Zhang, "A PID Controller Approach for Stochastic Optimization of Deep
Networks", CVPR 2018.
http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf

---
[Back to the Canon](../README.md)

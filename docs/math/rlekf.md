# RLEKF

Implements RLEKF, a reorganized-layer extended Kalman filter optimizer for training deep potential models.

RLEKF treats network training as a nonlinear state estimation problem: the weights are the hidden state and each label is a noisy measurement. The extended Kalman filter (EKF) updates the weights with a Kalman gain derived from an error covariance matrix $P_t$, and a memory (forgetting) factor $\lambda_t$ progressively discounts older observations. To make the full EKF tractable for large networks, RLEKF reorganizes the parameters into $L$ blocks and keeps a block-diagonal covariance, so the gain is computed per block from the local gradient.

The per-step EKF update is

$$
\begin{aligned}
a_t &= \lambda_t^{-1}\, H_t^{\top} P_{t-1} H_t + \alpha_t^2 R_t, \\
K_t &= \lambda_t^{-1}\, P_{t-1} H_t^{\top} a_t^{-1}, \\
P_t &= (I - K_t H_t)\, \lambda_t^{-1} P_{t-1}, \\
\theta_t &= \theta_{t-1} + K_t\, \varepsilon_t, \\
\varepsilon_t &= y_t - h(\theta_{t-1}, x_t), \\
\lambda_t &= 1 - (1 - \lambda_1)\, \nu^{\,t-1},
\end{aligned}
$$

where $\theta$ are the network weights, $g_t = H_t = \partial h(\theta, x_t)/\partial\theta$ is the Jacobian of the model output $h$ evaluated at $\theta_{t-1}$, $\varepsilon_t$ is the prediction error against label $y_t$, $K_t$ is the Kalman gain, $P_t$ is the weight error covariance ($P_0 = I$), $\alpha_t^2 R_t$ is the measurement noise covariance (set to $L I$), $\lambda_t \in (0,1]$ is the memory factor with initial value $\lambda_1$, and $\nu$ is the forgetting rate.

Reference: Siyu Hu, Wentao Zhang, Qiuchen Sha, Feng Pan, Lin-Wang Wang, Weile Jia, Guangming Tan, Tong Zhao, "RLEKF: An Optimizer for Deep Potential with Ab Initio Accuracy", AAAI 2023. https://arxiv.org/abs/2212.06989

---
[Back to the Canon](../index.md)

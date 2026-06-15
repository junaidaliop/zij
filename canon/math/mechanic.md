# MechanizedSGD

Implements Mechanic, a learning-rate tuner that wraps any base optimizer and automatically scales its updates.

Mechanic does not replace the base optimizer; it learns a scalar $s_t$ that multiplies the base optimizer's cumulative trajectory. The base optimizer emits an update $u_t$, and Mechanic accumulates the displacement $\Delta_{t+1} = \Delta_t + u_t$ from a fixed reference point $x_{\mathrm{ref}}$. A parameter-free online learner derived from coin-betting tunes the scale by treating the inner product of the cumulative displacement with the gradient as a one-dimensional "reward" signal.

The scale is maintained as $n$ parallel instances (default $n=6$), each driven by a different decay rate $\beta_i$, and their contributions are summed. The reward $r_t$ grows when the displacement aligns with descent, while the wealth $W_t$ converts accumulated reward into the next scale via a normalization by the running second moment.

$$
\begin{aligned}
h_t &= \left\langle \Delta_t,\; g_t + \lambda \Big(\textstyle\sum_i s_{t,i}\Big)\,\|g_t\|\,\frac{x_t}{\|x_t\|} \right\rangle \\
m_t &= \max(\beta \odot m_{t-1},\; h_t) \\
v_t &= \beta^2 \odot v_{t-1} + h_t^2 \\
r_t &= \max\!\big(0,\; \beta \odot r_{t-1} - s_{t-1} \odot h_t\big) \\
W_t &= \frac{s_{\mathrm{init}}\, m_t}{n} + r_t \\
s_{t+1} &= \frac{W_t}{\sqrt{v_t} + \epsilon} \\
x_{t+1} &= x_{\mathrm{ref}} + \Big(\textstyle\sum_i s_{t+1,i}\Big)\,\Delta_{t+1}
\end{aligned}
$$

where $g_t$ is the gradient, $\Delta_t$ is the cumulative base-optimizer displacement from the reference point $x_{\mathrm{ref}}$, $h_t$ is the scalar reward signal fed to the tuner, $\beta \in [0,1]^n$ are the per-instance decay rates, $m_t, v_t, r_t \in \mathbb{R}^n$ are the running max, second moment, and reward, $W_t$ is the wealth, $s_t \in \mathbb{R}^n$ is the vector of scale factors (the effective learning-rate multiplier is $\sum_i s_{t,i}$), $\lambda$ is the weight-decay coefficient, $s_{\mathrm{init}}$ is the initial scale, and $\epsilon$ is a stability constant. The operator $\odot$ denotes coordinate-wise (per-instance) products.

Reference: Ashok Cutkosky, Aaron Defazio, Harsh Mehta, "Mechanic: A Learning Rate Tuner", NeurIPS 2023. https://arxiv.org/abs/2306.00144

---
[Back to the Canon](../README.md)

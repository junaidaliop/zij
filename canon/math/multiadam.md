# MultiAdam

Implements MultiAdam, a scale-invariant Adam variant that balances multiple loss components in multiscale PINN training.

Physics-informed neural networks minimize a sum of heterogeneous loss terms (PDE residuals, boundary and initial conditions) whose gradient scales can differ by orders of magnitude, biasing standard Adam toward the dominant term. MultiAdam partitions the loss into $n$ groups and keeps an independent pair of moment estimates per group. Because the second moment tracks the gradient scale, each group's Adam-normalized step is rescaled to comparable magnitude before the steps are averaged, so no single loss dominates the update.

$$
\begin{aligned}
g_{t,i} &= \nabla_\theta f_i(\theta_{t-1}) \\
m_{t,i} &= \beta_1 m_{t-1,i} + (1-\beta_1) g_{t,i} \\
v_{t,i} &= \beta_2 v_{t-1,i} + (1-\beta_2) g_{t,i}^2 \\
\hat{m}_{t,i} &= \frac{m_{t,i}}{1-\beta_1^t}, \qquad \hat{v}_{t,i} = \frac{v_{t,i}}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\gamma}{n} \sum_{i=1}^{n} \frac{\hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}} + \epsilon}
\end{aligned}
$$

where $f_i$ is the $i$-th loss group, $g_{t,i}$ its gradient, $m_{t,i}/v_{t,i}$ the per-group first and second moments with bias-corrected forms $\hat{m}_{t,i}/\hat{v}_{t,i}$, $\gamma$ the learning rate, $\beta_1,\beta_2$ the decay rates, $n$ the number of groups, and $\epsilon$ a stability constant.

Reference: Jiachen Yao, Chang Su, Zhongkai Hao, Songming Liu, Hang Su, Jun Zhu, "MultiAdam: Parameter-wise Scale-invariant Optimizer for Multiscale Training of Physics-informed Neural Networks", ICML 2023. https://arxiv.org/abs/2306.02816

---
[Back to the Canon](../README.md)

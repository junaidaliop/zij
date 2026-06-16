# Lion

Implements Lion, a sign-based optimizer that tracks only momentum.

Lion (EvoLved Sign Momentum) was discovered through symbolic program search over the space of optimization algorithms. It keeps a single momentum buffer and applies the sign of an interpolated gradient–momentum direction, so every update has uniform magnitude across dimensions, which makes it more memory-efficient than adaptive methods that store second moments. The update direction and the momentum tracking use two different interpolation coefficients: $\beta_1$ controls the direction taken now, while $\beta_2$ controls how the momentum is rolled forward. Weight decay is decoupled.

$$
\begin{aligned}
c_t &= \mathrm{sign}\!\left(\beta_1 m_{t-1} + (1-\beta_1) g_t\right) \\
m_t &= \beta_2 m_{t-1} + (1-\beta_2) g_t \\
\theta_t &= \theta_{t-1} - \eta \left(c_t + \lambda \theta_{t-1}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the momentum, $\beta_1,\beta_2$ the interpolation coefficients (defaults $0.9$ and $0.99$), $\lambda$ the decoupled weight decay, and $\mathrm{sign}$ the element-wise sign.

Reference: Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le, "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023. https://arxiv.org/abs/2302.06675

---
[Back to the Canon](../index.md)

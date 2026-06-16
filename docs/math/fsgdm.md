# FSGDM

Implements FSGDM (Frequency Stochastic Gradient Descent with Momentum), SGD with a time-varying momentum coefficient derived from a frequency-domain view of momentum.

The paper analyzes the momentum recursion as a gradient filter and argues that the filtering should change over training: keep the original (high-frequency) gradient components early, then gradually amplify low-frequency components by raising the momentum coefficient toward 1. FSGDM realizes this with a monotonically increasing, stagewise-constant coefficient $u_t$ schedule on top of the standard heavy-ball momentum update.

$$
\begin{aligned}
u_t &= u\!\left(\left\lfloor t/\delta \right\rfloor \delta\right), \qquad u(t) = \frac{t}{t + \mu} \\
m_t &= u_t\, m_{t-1} + v\, g_t \\
\theta_t &= \theta_{t-1} - \gamma_t\, m_t
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma_t$ the (scheduled) learning rate, $g_t$ the stochastic gradient, $m_t$ the momentum buffer, $u_t$ the time-varying momentum coefficient, and $v$ a constant gradient coefficient ($v=1$). The schedule uses $\mu = c\,\Sigma$ with scaling factor $c$, total training steps $\Sigma$, stage length $\delta = \Sigma/N$, and number of stages $N$ (300 in the experiments); the floor makes $u_t$ piecewise-constant across stages.

Reference: Xianliang Li, Jun Luo, Zhiwei Zheng, Hanxiao Wang, Li Luo, Lingkun Wen, Linlong Wu, Sheng Xu, "On the Performance Analysis of Momentum Method: A Frequency Domain Perspective", ICLR 2025. https://arxiv.org/abs/2411.19671

---
[Back to the Canon](../index.md)

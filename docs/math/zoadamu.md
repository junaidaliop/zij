# ZO-AdaMU

Implements ZO-AdaMU, a zeroth-order optimizer that adapts the perturbation by simulated momentum and uncertainty.

ZO-AdaMU estimates gradients without backpropagation via simultaneous perturbation stochastic approximation (SPSA), evaluating the loss at $\theta \pm \epsilon m_t$. Unlike vanilla zeroth-order methods that perturb with a fresh zero-centered Gaussian, the perturbation $m_t$ is a momentum-perturbed Gaussian: a convex mix of a zero-centered draw and a draw centered at the previous momentum, with the mixing weights $\alpha_t$, $\beta_{1,t}$, $\beta_{2,t}$ annealed over a three-phase schedule so the perturbation interpolates from exploratory to momentum-aligned. A second moment $v_t$ formed from the squared perturbation components rescales the step adaptively.

$$
\begin{aligned}
\dot{z}_{t+1} &\sim \mathcal{N}(0,\ \sqrt{\alpha_{t+1}}) \\
\ddot{z}_{t+1} &\sim \mathcal{N}(m_t,\ \sqrt{1-\alpha_{t+1}}) \\
m_{t+1} &= \beta_{1}\,\dot{z}_{t+1} + (1-\beta_{1})\,\ddot{z}_{t+1} \\
\hat{g}_t &= \frac{\mathcal{L}(\theta_t+\epsilon m_t) - \mathcal{L}(\theta_t-\epsilon m_t)}{2\epsilon}\, m_t \\
v_{t+1} &= \beta_{2}\,\dot{z}_{t+1}^{2} + (1-\beta_{2})\,\ddot{z}_{t+1}^{2} \\
\theta_{t+1} &= \theta_t - \eta\,\frac{\hat{g}_t}{\sqrt{v_{t+1}^{2} + \sigma}}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\hat{g}_t$ the SPSA gradient estimate, $\epsilon$ the perturbation scale, $\dot{z}$ and $\ddot{z}$ the zero-centered and momentum-centered Gaussian components, $m_t$ the simulated-momentum perturbation, $v_t$ the second moment, $\alpha_t \in [0,1]$ the annealed uncertainty, $\beta_1,\beta_2 \in [0,1]$ the annealed smoothing weights, and $\sigma = 10^{-8}$ a stability constant.

Reference: Shuoran Jiang, Qingcai Chen, Youcheng Pan, Yang Xiang, Yukang Lin, Xiangping Wu, Chuanyi Liu, Xiaobao Song, "ZO-AdaMU Optimizer: Adapting Perturbation by the Momentum and Uncertainty in Zeroth-order Optimization", AAAI 2024. https://arxiv.org/abs/2312.15184

---
[Back to the Canon](../index.md)

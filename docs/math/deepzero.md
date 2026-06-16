# DeepZero

Implements DeepZero, a zeroth-order framework that trains deep networks from forward passes alone via coordinate-wise finite-difference gradient estimation.

DeepZero replaces backpropagation with a zeroth-order (ZO) gradient estimate built only from function evaluations. It uses the coordinate-wise gradient estimator (CGE), which perturbs each coordinate independently by a small smoothing step and forms a forward finite difference, in place of the higher-variance randomized estimator (RGE). To make this scalable, the estimate is restricted to a sparse active coordinate set $\mathcal{S}$ found by ZO-GraSP pruning, and the resulting estimate drives a standard SGD update.

$$
\begin{aligned}
\hat{g}_t &= \sum_{i \in \mathcal{S}} \frac{\ell(\theta_t + \mu e_i) - \ell(\theta_t)}{\mu}\, e_i \\
\theta_{t+1} &= \theta_t - \eta\, \hat{g}_t
\end{aligned}
$$

where $\hat{g}_t$ is the CGE of $\nabla \ell$, $e_i$ is the $i$-th standard basis vector, $\mu > 0$ is the perturbation (smoothing) size, $\mathcal{S}$ is the active coordinate set (full coordinate set $\{1,\dots,d\}$ in the dense case), and $\eta$ is the learning rate.

Reference: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu, "DeepZero: Scaling Up Zeroth-Order Optimization for Deep Model Training", ICLR 2024. https://arxiv.org/abs/2310.02025

---
[Back to the Canon](../index.md)

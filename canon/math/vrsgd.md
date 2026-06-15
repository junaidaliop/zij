# VR-SGD

Implements VR-SGD, a stochastic variance-reduced method that decouples the snapshot point from the epoch starting point.

VR-SGD runs in epochs. Once per epoch it computes a full gradient $\mu_s$ at a snapshot $\tilde\theta_{s-1}$, then performs $m$ inner steps using the SVRG-style variance-reduced estimator. Its defining feature is that the snapshot point and the starting point of each epoch are chosen differently: the snapshot is the **average** of the previous epoch's iterates, while the next epoch **starts from the last iterate**. This decoupling, together with a learning rate that can be increased over early epochs, lets VR-SGD use a much larger step size than SVRG.

$$
\begin{aligned}
\mu_s &= \tfrac{1}{n}\sum_{i=1}^{n}\nabla f_i(\tilde\theta_{s-1}) \\
\tilde g_k &= \nabla f_{i_k}(\theta_k) - \nabla f_{i_k}(\tilde\theta_{s-1}) + \mu_s \\
\theta_{k+1} &= \theta_k - \eta_s\big[\tilde g_k + \nabla g(\theta_k)\big] \\
\tilde\theta_{s} &= \tfrac{1}{m}\sum_{k=1}^{m}\theta_k, \qquad \theta_0^{(s+1)} = \theta_m^{(s)}
\end{aligned}
$$

where $\theta_k$ are the inner iterates of epoch $s$, $\tilde\theta_{s-1}$ is the snapshot (the average over the previous epoch), $\mu_s$ is the full gradient at that snapshot, $i_k$ is a uniformly sampled index, $\eta_s$ is the per-epoch learning rate (optionally grown via $\eta_s = \eta_0/\max\{\alpha,\,2/(s+1)\}$ with $\alpha\in(0,1]$), $n$ is the number of components, $m$ the inner iterations per epoch, and $g$ a possibly non-smooth regularizer (the smooth case drops $\nabla g$).

Reference: Fanhua Shang, Kaiwen Zhou, Hongying Liu, James Cheng, Ivor W. Tsang, Lijun Zhang, Dacheng Tao, Licheng Jiao, "VR-SGD: A Simple Stochastic Variance Reduction Method for Machine Learning", arXiv 2018. https://arxiv.org/abs/1802.09932

---
[Back to the Canon](../README.md)

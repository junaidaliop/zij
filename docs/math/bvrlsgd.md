# BVR-L-SGD

Implements BVR-L-SGD, bias-and-variance reduced local SGD for heterogeneous federated learning.

Each communication round begins with a (large-batch) snapshot gradient computed by every worker $p$ at the shared point $\tilde\theta$; these are averaged across workers to seed a SARAH-style recursive estimator. Each worker then runs $K$ local steps, recursively correcting its gradient estimate with the difference of stochastic gradients at consecutive iterates, which drives the variance to zero. To reduce the bias introduced by data heterogeneity, synchronization does not average the local models: instead one worker is selected uniformly at random and its iterate becomes the next round's shared point.

$$
\begin{aligned}
v_0^{(p)} &= \frac{1}{P}\sum_{q=1}^{P} \nabla f_q(\tilde\theta), \qquad \theta_0^{(p)} = \tilde\theta, \\
g_k^{(p)}(\theta) &= \frac{1}{b}\sum_{z \in \mathcal{B}_k^{(p)}} \nabla \ell(\theta, z), \\
v_k^{(p)} &= g_k^{(p)}\!\big(\theta_{k-1}^{(p)}\big) - g_k^{(p)}\!\big(\theta_{k-2}^{(p)}\big) + v_{k-1}^{(p)}, \\
\theta_k^{(p)} &= \theta_{k-1}^{(p)} - \eta\, v_k^{(p)}, \\
\tilde\theta &\leftarrow \theta_K^{(\hat p)}, \qquad \hat p \sim \mathrm{Unif}[P].
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $P$ the number of workers, $f_q$ the local objective on worker $q$, $\mathcal{B}_k^{(p)}$ a minibatch of size $b$ drawn by worker $p$ at step $k$, $g_k^{(p)}$ its stochastic gradient, $v_k^{(p)}$ the bias-and-variance reduced gradient estimator, $K$ the number of local steps per round, and $\hat p$ a uniformly randomly selected worker whose iterate replaces the average at synchronization.

Reference: Tomoya Murata, Taiji Suzuki, "Bias-Variance Reduced Local SGD for Less Heterogeneous Federated Learning", ICML 2021. https://arxiv.org/abs/2102.03198

---
[Back to the Canon](../index.md)

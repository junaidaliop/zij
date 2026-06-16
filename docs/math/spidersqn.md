# SpiderSQN

Implements SpiderSQN, a stochastic quasi-Newton method that drives a damped L-BFGS direction with a SPIDER variance-reduced gradient.

SpiderSQN combines two ingredients. The gradient estimator $v_k$ is built with the SPIDER recursion: every $q$ steps it is refreshed with a full gradient, and in between it is updated by accumulating stochastic gradient differences, which keeps its variance small without recomputing the full gradient each step. That low-variance estimator is then turned into a search direction $d_k = H_k v_k$, where $H_k$ is the inverse-Hessian approximation produced by stochastic damped L-BFGS (SdLBFGS) via two-loop recursion over stored curvature pairs. The parameter step is a fixed-rate move along $-d_k$.

$$
\begin{aligned}
v_k &= \begin{cases} \nabla f(x_k) & \text{if } \mathrm{mod}(k, q) = 0 \\ \nabla f_{\xi_k}(x_k) - \nabla f_{\xi_k}(x_{k-1}) + v_{k-1} & \text{otherwise} \end{cases} \\
d_k &= H_k v_k \\
x_{k+1} &= x_k - \eta\, d_k
\end{aligned}
$$

where $x$ are the parameters, $\eta$ the step size, $f$ the finite-sum objective with components $f_i$, $\xi_k$ a sampled index (or minibatch) so that $\nabla f_{\xi_k}$ is a stochastic gradient, $q$ the period between full-gradient refreshes, $v_k$ the SPIDER gradient estimator, and $H_k$ the SdLBFGS inverse-Hessian approximation applied to $v_k$ through the L-BFGS two-loop recursion.

Reference: Qingsong Zhang, Feihu Huang, Cheng Deng, Heng Huang, "Faster Stochastic Quasi-Newton Methods", arXiv 2020. https://arxiv.org/abs/2004.06479

---
[Back to the Canon](../index.md)

# FedIvon

Implements FedIvon, federated learning where each client runs the Improved Variational Online Newton (IVON) optimizer and the server merges clients by Fisher-weighted averaging.

Each client maintains a diagonal Gaussian posterior over the weights with mean $\mu$ and Hessian (precision) estimate $h$. Locally it runs IVON: parameters are sampled $\theta \sim \mathcal{N}(\mu, \sigma^2)$ with $\sigma^2 = 1/(\lambda(h+\delta))$, the gradient at the sample yields a reparameterization estimate of the per-coordinate Hessian, and momentum-smoothed gradient and curvature drive a Newton-style mean update. After local training the server treats each client's $h$ as a Fisher precision and performs a precision-weighted merge of the means, which doubles as a natural model-averaging rule and provides posterior uncertainty.

$$
\begin{aligned}
\theta &\sim \mathcal{N}\!\left(\mu,\ \tfrac{1}{\lambda(h+\delta)}\right), \quad \hat g = \nabla \ell(\theta), \quad \hat h = \hat g \cdot \frac{\theta-\mu}{\sigma^2} \\
g_t &= \beta_1 g_{t-1} + (1-\beta_1)\hat g \\
h_t &= \beta_2 h_{t-1} + (1-\beta_2)\hat h + \tfrac{1}{2}(1-\beta_2)^2 \frac{(h_{t-1}-\hat h)^2}{h_{t-1}+\delta} \\
\bar g_t &= \frac{g_t}{1-\beta_1^{\,t}}, \qquad \mu_t = \mu_{t-1} - \eta\,\frac{\bar g_t + \delta\,\mu_{t-1}}{h_t + \delta} \\
\text{(server)}\quad h^{(r+1)} &= \sum_{k} w_k\, h_k, \qquad \mu^{(r+1)} = \frac{\sum_{k} w_k\, h_k \odot \mu_k}{\sum_{k} w_k\, h_k}
\end{aligned}
$$

where $\mu$ is the variational mean (the deployed weights $\theta$), $\sigma^2$ the posterior variance, $h_t$ the diagonal Hessian/precision estimate, $g_t$ the momentum, $\hat g$ and $\hat h$ the sampled gradient and reparameterization Hessian, $\eta$ the learning rate, $\beta_1,\beta_2$ the momentum decays, $\delta$ the weight decay (prior precision), $\lambda$ the effective sample size, $\odot$ element-wise product, and $w_k = N_k / \sum_j N_j$ the data-proportion weight of client $k$ over communication round $r$.

Reference: Shivam Pal, Aishwarya Gupta, Saqib Sarwar, Piyush Rai, "Federated Learning with Uncertainty and Personalization via Efficient Second-order Optimization", TMLR 2025. https://arxiv.org/abs/2411.18385

---
[Back to the Canon](../README.md)

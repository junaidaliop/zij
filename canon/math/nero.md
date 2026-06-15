# Nero

Implements Nero, a neural-architecture-aware optimizer that constrains each neuron's weights to the unit hypersphere with zero mean.

Nero acts per neuron. For a neuron with weight vector $w$ and bias $b$, it tracks a running average of the squared gradient norm and rescales the update so that each neuron is normalized by its own gradient magnitude. The learning rate for $w$ is scaled by $\|w\|_2$, and after each step $w$ is projected back onto the balanced constraint set: zero mean and unit $\ell_2$ norm. Biases use their initialization scale $\sigma_b$ in place of the norm. The running averages are bias-corrected as in Adam.

$$
\begin{aligned}
\bar{g}^2_{w,t} &= \beta\,\bar{g}^2_{w,t-1} + (1-\beta)\,\|g_{w,t}\|_2^2, \\
\bar{g}^2_{b,t} &= \beta\,\bar{g}^2_{b,t-1} + (1-\beta)\,g_{b,t}^2, \\
\hat{g}_{w,t} &= \sqrt{\bar{g}^2_{w,t} / (1-\beta^t)}, \qquad \hat{g}_{b,t} = \sqrt{\bar{g}^2_{b,t} / (1-\beta^t)}, \\
w_t &\leftarrow w_{t-1} - \eta\,\frac{\|w_{t-1}\|_2}{\hat{g}_{w,t}}\,g_{w,t}, \\
b_t &\leftarrow b_{t-1} - \eta\,\frac{\sigma_b}{\hat{g}_{b,t}}\,g_{b,t}, \\
w_t &\leftarrow w_t - \tfrac{1}{n}\textstyle\sum_{i=1}^{n} w_{t,i}, \qquad w_t \leftarrow w_t / \|w_t\|_2.
\end{aligned}
$$

where $w$ and $b$ are a single neuron's weight vector (length $n$) and bias, $g_{w,t}$ and $g_{b,t}$ their gradients, $\bar{g}^2$ the running average of the squared gradient (norm), $\hat{g}$ its bias-corrected square root, $\eta$ the learning rate, $\beta$ the averaging decay, and $\sigma_b$ the initialization scale of the bias. The final two lines project $w$ to zero mean and unit norm.

Reference: Yang Liu, Jeremy Bernstein, Markus Meister, Yisong Yue, "Learning by Turning: Neural Architecture Aware Optimisation", ICML 2021. https://arxiv.org/abs/2102.07227

---
[Back to the Canon](../README.md)

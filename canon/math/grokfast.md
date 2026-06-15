# Grokfast

Implements Grokfast, a gradient filter that accelerates grokking by amplifying the slow-varying component of the gradient.

The optimization trajectory's gradient is decomposed into a fast-varying (overfitting) component and a slow-varying (generalizing) component. Grokfast applies a low-pass filter to the gradient sequence and adds the filtered slow component back, scaled by $\lambda$, before handing the result to any base optimizer (SGD, Adam, AdamW). The recommended variant uses an exponential moving average as the filter.

$$
\begin{aligned}
\mu_t &= \alpha\,\mu_{t-1} + (1-\alpha)\,g_t \\
\hat{g}_t &= g_t + \lambda\,\mu_t \\
\theta_{t+1} &= \mathrm{Opt}(\theta_t, \hat{g}_t)
\end{aligned}
$$

where $g_t$ is the raw gradient, $\mu_t$ is the exponential moving average of past gradients with momentum $\alpha \in [0,1)$, $\lambda$ is the amplifier scaling the slow component, $\hat{g}_t$ is the filtered gradient passed to the base optimizer $\mathrm{Opt}$, and $\theta$ are the parameters. The alternative Grokfast-MA replaces $\mu_t$ with the windowed mean $\mathrm{Avg}(Q)$ over a fixed-capacity queue $Q$ of the last $w$ gradients, giving $\hat{g}_t = g_t + \lambda\,\mathrm{Avg}(Q)$.

Reference: Jaerin Lee, Bong Gyun Kang, Kihoon Kim, Kyoung Mu Lee, "Grokfast: Accelerated Grokking by Amplifying Slow Gradients", arXiv 2024. https://arxiv.org/abs/2405.20233

---
[Back to the Canon](../README.md)

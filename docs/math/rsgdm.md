# RSGDM

Implements RSGDM, SGD with momentum corrected by a differential (gradient-change) estimate.

SGDM estimates the overall gradient with an exponential moving average of $g_t$, which is biased and lags behind the true gradient. RSGDM additionally tracks the exponential moving average of the gradient difference $\Delta g_t = g_t - g_{t-1}$ and adds it, scaled by $\beta$, to the usual momentum. This differential term corrects the bias and reduces the lag without introducing any new hyperparameter.

$$
\begin{aligned}
\Delta g_t &= g_t - g_{t-1} \\
m_t &= \beta m_{t-1} + (1 - \beta) g_t \\
z_t &= \beta z_{t-1} + (1 - \beta) \Delta g_t \\
n_t &= m_t + \beta z_t \\
\theta_t &= \theta_{t-1} - \eta\, n_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the gradient, $\beta$ is the momentum decay, $m_t$ is the gradient EMA, $z_t$ is the EMA of the gradient difference $\Delta g_t$, and $n_t$ is the differentially corrected gradient estimate used in the update.

Reference: Honglin Qin, Hongye Zheng, Bingxing Wang, Zhizhong Wu, Bingyao Liu, Yuanfang Yang, "Reducing Bias in Deep Learning Optimization: The RSGDM Approach", arXiv 2024. https://arxiv.org/abs/2409.15314

---
[Back to the Canon](../index.md)

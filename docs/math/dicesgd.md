# DiceSGD

Implements DiceSGD, differentially private SGD that removes clipping bias via error feedback.

Standard DP-SGD clips each per-sample gradient and adds Gaussian noise, but the clipping introduces a bias that does not vanish as training proceeds. DiceSGD keeps an error-feedback state $e_t$ that accumulates the discrepancy between the true minibatch gradient and the privatized update direction. The accumulated error is itself clipped (with a larger threshold $C_2 \ge C_1$) and fed back into the next step, so that the bias is corrected over time while the sensitivity stays bounded for the privacy guarantee.

$$
\begin{aligned}
v_t &= \frac{1}{B}\sum_{i\in B_t}\mathrm{clip}\!\big(g_t^{(i)},\, C_1\big) + \mathrm{clip}\!\big(e_t,\, C_2\big), \\
\theta_{t+1} &= \theta_t - \eta_t\,(v_t + w_t),\qquad w_t \sim \mathcal{N}(0,\, \sigma^2 I), \\
e_{t+1} &= e_t + \frac{1}{B}\sum_{i\in B_t} g_t^{(i)} - v_t,\qquad e_0 = 0,
\end{aligned}
$$

where $\mathrm{clip}(v, C) = \min\{1,\, C/\lVert v\rVert\}\,v$, $g_t^{(i)}$ is the per-sample gradient, $B$ is the batch size, $C_1$ clips individual gradients, $C_2 \ge C_1$ clips the error-feedback state, $\eta_t$ is the learning rate, and $w_t$ is Gaussian noise with variance $\sigma^2$ set by the privacy budget $(\varepsilon, \delta)$.

Reference: Xinwei Zhang, Zhiqi Bu, Zhiwei Steven Wu, Mingyi Hong, "Differentially Private SGD Without Clipping Bias: An Error-Feedback Approach", ICLR 2024. https://arxiv.org/abs/2311.14632

---
[Back to the Canon](../index.md)

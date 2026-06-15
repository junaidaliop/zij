# WSAM

Implements WSAM, sharpness-aware minimization with the sharpness weighted as a regularization term.


$$
\begin{aligned}
     &L^{\mathit{WSAM}}(\theta) = L(\theta) + \frac{\gamma}{1 - \gamma}
         \Bigl( \max_{\lVert \delta \rVert_2 \leq \rho} L(\theta + \delta)
         - L(\theta) \Bigr)                                               \\
     &\delta_t = \rho \, \frac{g_t}{\lVert g_t \rVert_2 + \epsilon}, \qquad
      \tilde{g}_t = \nabla L(\theta_t + \delta_t)                         \\
     &\theta_{t+1} = \theta_t - \eta \, \Bigl( u_t + \frac{\gamma}{1 - \gamma}
         \bigl( \tilde{g}_t - g_t \bigr) \Bigr)
\end{aligned}
$$

where $u_t$ is the base optimizer update computed from $g_t$.
With `decouple=False` the weighted gradient
$g_t + \frac{\gamma}{1 - \gamma} (\tilde{g}_t - g_t)$ is fed to the
base optimizer instead.

Reference: Yun Yue, Jiadi Jiang, Zhiling Ye, Ning Gao, Yongchao Liu, Ke Zhang,
"Sharpness-Aware Minimization Revisited: Weighted Sharpness as a Regularization Term", KDD 2023.
https://arxiv.org/abs/2305.15817


**Note:** WSAM wraps a base optimizer and needs two forward-backward passes per step: call `step` with a closure, or call `first_step` and `second_step` around the second backward pass. Pass `model` so BatchNorm running stats are frozen during the second pass.


---
[Back to the Canon](../README.md)

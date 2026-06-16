# ASAM

Implements ASAM, adaptive sharpness-aware minimization.


$$
\begin{aligned}
&g_t = \nabla L(\theta_t), \qquad
 \hat{\epsilon}_t = \rho \, \frac{T_{\theta_t}^2 \, g_t}
     {\lVert T_{\theta_t} \, g_t \rVert_2}                       \\
&\theta_{t+1} = \theta_t - \eta \, \nabla L(\theta)
    \big\rvert_{\theta = \theta_t + \hat{\epsilon}_t}
\end{aligned}
$$

where $T_{\theta_t} = \mathrm{diag}(\lvert \theta_t \rvert)$
is the normalization operator that makes the maximization region
scale-invariant, and the gradient at the perturbed point is fed to the
wrapped base optimizer. This is `SAM` with `adaptive=True`; the
`rho=2.0` default follows the community davda54/sam implementation
(~10x SAM's radius for adaptive mode); the paper tunes rho per task in
the 0.2-1.0 range.

Reference: Jungmin Kwon, Jeongseop Kim, Hyunseo Park, In Kwon Choi,
"ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning
of Deep Neural Networks", ICML 2021.
https://arxiv.org/abs/2102.11600


**Note:** This class follows davda54/sam's adaptive mode rather than the authors' official repo. Each step needs two forward-backward passes, as with `SAM`.


---
[Back to the Canon](../index.md)

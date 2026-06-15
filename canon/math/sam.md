# SAM

Implements SAM, sharpness-aware minimization wrapping a base optimizer.


$$
\begin{aligned}
&g_t = \nabla L(\theta_t), \qquad
 \hat{\epsilon}_t = \rho \, \frac{g_t}{\lVert g_t \rVert_2}      \\
&\theta_{t+1} = \theta_t - \eta \, \nabla L(\theta)
    \big\rvert_{\theta = \theta_t + \hat{\epsilon}_t}
\end{aligned}
$$

where $\hat{\epsilon}_t$ solves the inner maximization
$\max_{\lVert \epsilon \rVert_2 \leq \rho} L(\theta_t + \epsilon)$
to first order, and the gradient at the perturbed point is fed to the
wrapped base optimizer. With `adaptive=True` the perturbation becomes
the scale-invariant
$\hat{\epsilon}_t = \rho \, \theta_t^2 g_t / \lVert \theta_t g_t \rVert_2$
of ASAM (Kwon et al., ICML 2021).

Reference: Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur,
"Sharpness-Aware Minimization for Efficiently Improving Generalization",
ICLR 2021.
https://arxiv.org/abs/2010.01412


**Note:** Each step needs two forward-backward passes: either call `first_step`, recompute the loss and gradients, then call `second_step`, or pass `step` a closure that zeroes gradients, computes the loss, and calls `backward()`.


---
[Back to the Canon](../README.md)

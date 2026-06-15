# LookSAM

Implements LookSAM, a Sharpness-Aware Minimization variant that takes the ascent step only once every $k$ steps.


$$
\begin{aligned}
&\text{if } t \bmod k = 0: \\
&\quad \epsilon_t = \rho \, \frac{g_t}{\lVert g_t \rVert_2}, \qquad
   g_s = \nabla_{\theta} L(\theta_t + \epsilon_t) \\
&\quad g_v = g_s - \lVert g_s \rVert_2
   \frac{g_t \cdot g_s}{\lVert g_t \rVert_2 \lVert g_s \rVert_2}
   \frac{g_t}{\lVert g_t \rVert_2} \\
&\text{otherwise:} \\
&\quad g_s = g_t + \alpha
   \frac{\lVert g_t \rVert_2}{\lVert g_v \rVert_2} g_v \\
&\theta_{t+1} = \theta_t - \eta \, g_s
\end{aligned}
$$

where $g_t$ is the minibatch gradient at $\theta_t$ and the
descent update with $g_s$ is delegated to the wrapped base optimizer.


**Note:** Gradients must be computed before calling `step`, which takes a closure that re-evaluates the loss and calls `backward()` at the perturbed point; the closure runs only on refresh steps (`get_step() % k == 0`). Alternatively, call `first_step` and `second_step` explicitly, running the second forward-backward pass only on refresh steps. Following the upstream implementation, the $g_v$ decomposition and the reuse-step scaling are applied per parameter tensor rather than over the global parameter vector. The refresh schedule reads a `step` counter from the base optimizer, so the base should be an optimizer that tracks per-step state such as Adam or AdamW; with a stateless base every step refreshes, which is correct but saves no computation.

Reference: Yong Liu, Siqi Mai, Xiangning Chen, Cho-Jui Hsieh, Yang You,
"Towards Efficient and Scalable Sharpness-Aware Minimization", CVPR 2022.
https://arxiv.org/abs/2203.02714

---
[Back to the Canon](../README.md)

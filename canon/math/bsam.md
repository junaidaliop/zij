# bSAM

Implements bSAM, a Bayesian, Adam-like extension of Sharpness-Aware Minimization.

The paper shows that SAM is the optimal convex relaxation of the Bayes objective, obtained through the Fenchel biconjugate of the expected loss. This view turns SAM into a variational method over a Gaussian posterior $\mathcal{N}(\boldsymbol{\omega}, \boldsymbol{\sigma}^2)$, where the inverse variance is tracked by a per-parameter scale vector $\mathbf{s}$ with $\boldsymbol{\sigma}^2 = 1/(N\mathbf{s})$. The resulting optimizer perturbs the weights with a preconditioned ascent step, then performs an Adam-style mean update while estimating the scale from gradient magnitudes, yielding uncertainty estimates alongside the trained weights.

At each step bSAM samples $\boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{\omega}, \boldsymbol{\sigma}^2)$, evaluates the perturbed gradient $\mathbf{g}_\epsilon$ at $\boldsymbol{\omega}+\boldsymbol{\epsilon}$, and updates:

$$
\begin{aligned}
\boldsymbol{\epsilon} &\leftarrow \rho\,\frac{\mathbf{g}}{\mathbf{s}} \\
\mathbf{g}_m &\leftarrow \beta_1 \mathbf{g}_m + (1-\beta_1)\bigl(\mathbf{g}_\epsilon + \delta\boldsymbol{\omega}\bigr) \\
\mathbf{s} &\leftarrow \beta_2\,\mathbf{s} + (1-\beta_2)\bigl(\sqrt{\mathbf{s}}\cdot|\mathbf{g}| + \delta + \gamma\bigr) \\
\boldsymbol{\omega} &\leftarrow \boldsymbol{\omega} - \alpha\,\frac{\mathbf{g}_m}{\mathbf{s}}
\end{aligned}
$$

where $\boldsymbol{\omega}$ is the posterior mean, $\mathbf{s}$ the per-parameter scale (inverse variance $\boldsymbol{\sigma}^2 = 1/(N\mathbf{s})$ with $N$ the dataset size), $\mathbf{g}$ the gradient at the sampled $\boldsymbol{\theta}$, $\mathbf{g}_\epsilon$ the gradient at the perturbed point $\boldsymbol{\omega}+\boldsymbol{\epsilon}$, $\mathbf{g}_m$ the momentum, $\alpha$ the learning rate, $\rho$ the SAM perturbation radius, $\delta$ the $L_2$ regularizer, $\gamma$ a damping constant, and $\beta_1,\beta_2$ the momentum and scale decay rates. All vector operations are elementwise.

Reference: Thomas Möllenhoff, Mohammad Emtiyaz Khan, "SAM as an Optimal Relaxation of Bayes", ICLR 2023. https://arxiv.org/abs/2210.01620

---
[Back to the Canon](../README.md)

# AGZO

Implements AGZO, an activation-guided zeroth-order optimizer that confines perturbations to a low-rank activation subspace.

AGZO fine-tunes language models with forward passes only, in the spirit of MeZO, but replaces the isotropic Gaussian perturbation with a structured one. For each linear layer it first extracts an orthonormal basis $A_\ell$ spanning the dominant directions of the layer's activation matrix $H_\ell$ via a few steps of power iteration, then samples the perturbation inside that subspace as $\Delta_\ell = R_\ell A_\ell^\top$. Nonlinear layers fall back to a plain Gaussian perturbation. Steering the probe directions toward where activations actually have energy reduces the variance of the gradient estimate.

The gradient is the one-sided forward-difference quotient: perturb in place by $\mu \Delta_\ell$, evaluate the perturbed loss $f_+$, and reconstruct a rank-structured gradient estimate by scaling $\Delta_\ell$ with the scalar $g = (f_+ - f_0)/\mu$. The descent step both undoes the perturbation and applies the update, so only the single scalar $g$ and the random seeds need to be carried between evaluations.

$$
\begin{aligned}
\Delta_\ell &= R_\ell A_\ell^\top \;\;(\text{linear}), \qquad \Delta_\ell = u_\ell \;\;(\text{nonlinear}), \qquad R_\ell, u_\ell \sim \mathcal{N}(0, I) \\
g &= \frac{f(W + \mu \Delta_\ell; B) - f(W; B)}{\mu} = \frac{f_+ - f_0}{\mu} \\
\widehat{\nabla}_{W_\ell} f^{\mathrm{AGZO}}(W; B) &= g\, \Delta_\ell \\
W_\ell &\leftarrow W_\ell - \mu \Delta_\ell - \eta\, g\, \Delta_\ell
\end{aligned}
$$

where $W_\ell$ are the weights of layer $\ell$, $f(W; B)$ is the loss on minibatch $B$, $A_\ell$ is the orthonormal activation-informed basis from power iteration on $H_\ell H_\ell^\top$, $R_\ell$ and $u_\ell$ are i.i.d. standard Gaussian, $\mu > 0$ is the perturbation scale, $\eta$ is the learning rate, $f_0$ and $f_+$ are the unperturbed and perturbed losses, the term $-\mu \Delta_\ell$ restores the parameters and $-\eta\, g\, \Delta_\ell$ applies the step.

Reference: Wei Lin, Yining Jiang, Qingyu Song, Qiao Xiang, Hong Xu, "AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning", ICML 2026. https://arxiv.org/abs/2601.17261

---
[Back to the Canon](../README.md)

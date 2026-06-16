# Lookbehind-SAM

Implements Lookbehind-SAM, a Sharpness-Aware Minimization variant that takes $k$ inner ascent steps to build a stronger perturbation, then a single interpolated descent step.

Standard SAM uses one gradient ascent step to locate an adversarial neighbor and one descent step from it. Lookbehind unrolls the ascent into $k$ steps, accumulating the perturbation iteratively so the worst-case point is sharper, while the descent gradients gathered along the way are combined through a Lookahead-style linear interpolation between the starting weights and the final fast weights. This trades a few extra gradient evaluations per iteration for a tighter sharpness estimate and reduced variance.

$$
\begin{aligned}
\phi_{t,0} &= \phi_{t-1}, \quad \phi'_{t,0} = \phi_{t-1}, \\
\epsilon_{t,i} &= \rho\,\frac{\nabla\mathcal{L}_{\mathcal{D}}(\phi'_{t,i-1})}{\|\nabla\mathcal{L}_{\mathcal{D}}(\phi'_{t,i-1})\|_2}, \\
\phi'_{t,i} &= \phi'_{t,i-1} + \epsilon_{t,i}, \\
\phi_{t,i} &= \phi_{t,i-1} - \eta\,\nabla\mathcal{L}_{\mathcal{D}}(\phi'_{t,i}), \qquad i = 1,\dots,k, \\
\phi_t &= \phi_{t-1} + \alpha\,(\phi_{t,k} - \phi_{t-1}),
\end{aligned}
$$

where $\phi$ are the parameters, $\eta$ the inner descent step size, $\rho$ the SAM neighborhood radius, $\epsilon_{t,i}$ the perturbation at inner step $i$, $\phi'_{t,i}$ the perturbed (ascending) weights, $\phi_{t,i}$ the fast (descending) weights, $k$ the number of inner steps, $\alpha \in (0,1]$ the outer interpolation coefficient, and $\mathcal{L}_{\mathcal{D}}$ the loss over training set $\mathcal{D}$.

Reference: Gonçalo Mordido, Pranshu Malviya, Aristide Baratin, Sarath Chandar, "Lookbehind-SAM: k steps back, 1 step forward", ICML 2024. https://arxiv.org/abs/2307.16704

---
[Back to the Canon](../index.md)

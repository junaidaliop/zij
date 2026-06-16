# KO

Implements KO, a kinetics-inspired wrapper that perturbs gradients via simulated particle collisions before a base optimizer step.

KO (Kinetics-inspired Optimizer) views the rows of a weight matrix as colliding particles and reformulates training as the evolution of a particle system. To counteract parameter condensation and promote diversity, it injects a repulsion force into the gradient: neuron pairs that are similar in both weight and gradient direction repel one another. The modified gradient is then handed to a standard base optimizer (SGD or Adam), so KO adds no parameters of its own beyond a single collision coefficient.

The practical "soft collision" form perturbs the layer gradient, then applies the base update:

$$
\begin{aligned}
\Delta g_t &= -\,\cos(w,w)\,\cos(g_t,g_t)\,g_t, \\
\tilde g_t &= g_t + \gamma_c\,\Delta g_t, \\
\theta_{t+1} &= \mathrm{BaseOpt}(\theta_t, \tilde g_t).
\end{aligned}
$$

where $w \in \mathbb{R}^{N\times D}$ and $g_t \in \mathbb{R}^{N\times D}$ are the layer weight and gradient matrices, $\cos(w,w)$ and $\cos(g_t,g_t)$ are the $N\times N$ pairwise cosine-similarity matrices of the weight rows and gradient rows, $\gamma_c$ is the collision coefficient controlling the fraction of colliding particles, and $\mathrm{BaseOpt}$ is the unmodified SGD or Adam update applied to the collision-adjusted gradient $\tilde g_t$.

Reference: Mingquan Feng, Yixin Huang, Yifan Fu, Shaobo Wang, Junchi Yan, "KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches", arXiv 2025. https://arxiv.org/abs/2505.14777

---
[Back to the Canon](../index.md)

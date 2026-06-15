# ⊥Grad (OrthoGrad)

Implements ⊥Grad (OrthoGrad), a gradient projection that removes the component of the gradient aligned with the current weights.

Introduced to study grokking at the edge of numerical stability, ⊥Grad targets Naïve Loss Minimization, where loss decreases purely by scaling up the weight norm without changing predictions. Before the base optimizer step, it projects the gradient onto the hyperplane orthogonal to the current weight vector, keeping only the component that can alter the model's outputs. The projection wraps an existing optimizer (the paper uses ⊥AdamW and ⊥SGD), which then proceeds on the orthogonalized gradient.

$$
\begin{aligned}
g_t^{\perp} &= g_t - \frac{\theta_t^{\top} g_t}{\theta_t^{\top} \theta_t}\,\theta_t \\
\theta_{t+1} &= \theta_t - \eta\, g_t^{\perp}
\end{aligned}
$$

where $\theta_t$ are the parameters, $g_t = \nabla \mathcal{L}(\theta_t)$ the gradient, $g_t^{\perp}$ its component orthogonal to $\theta_t$, and $\eta$ the learning rate.

Reference: Lucas Prieto, Melih Barsbey, Pedro A.M. Mediano, Tolga Birdal, "Grokking at the Edge of Numerical Stability", arXiv 2025. https://arxiv.org/abs/2501.04697

---
[Back to the Canon](../README.md)

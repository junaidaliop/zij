# L2O-CFGD

Implements L2O-CFGD, a learned optimizer that drives Caputo fractional gradient descent by predicting its hyperparameters with a recurrent meta-optimizer.

Caputo fractional gradient descent steps along a fractional derivative anchored at a terminal point $c$, which embeds a tunable memory of the loss landscape between $c$ and the current iterate. Its behavior is sensitive to three quantities that are awkward to set by hand: the per-coordinate fractional order $\alpha$, a smoothing term $\beta$ that mixes in the next-order derivative, and the anchor $c$. L2O-CFGD replaces this hand-tuning with a learned-to-optimize approach: a recurrent network $M$ (parameters $\varphi$) reads the ordinary gradient and its own hidden state and emits $\alpha$, $\beta$, and $c$ at every step, after which a standard descent step is taken along the resulting scaled Caputo fractional gradient (computed in practice by Gauss-Jacobi quadrature).

$$
\begin{aligned}
[\alpha^{(t)}, \beta^{(t)}, c^{(t)}, h^{(t+1)}] &= M\!\left(\nabla_\theta f(\theta^{(t)}),\, h^{(t)},\, \varphi\right) \\
g^{(t)} &= {}_{c^{(t)}} D^{\alpha^{(t)}}_{\beta^{(t)}} f(\theta^{(t)}) \\
\theta^{(t+1)} &= \theta^{(t)} - \eta^{(t)}\, g^{(t)}
\end{aligned}
$$

where the scaled Caputo fractional gradient is

$$
{}_{c} D^{\alpha}_{\beta} f(\theta) = \mathrm{diag}\!\left({}_{c_j}^{C} \mathcal{D}^{\alpha_j}_{\theta_j} I(\theta_j)\right)^{-1}\!\left[{}_{c}^{C}\nabla^{\alpha}_{\theta} f(\theta) + \beta\cdot \mathrm{diag}\!\left(|\theta_j - c_j|\right)\, {}_{c}^{C}\nabla^{1+\alpha}_{\theta} f(\theta)\right]
$$

where $\theta$ are the parameters, $\eta^{(t)}$ the learning rate, $\nabla_\theta f$ the ordinary gradient, ${}_{c}^{C}\nabla^{\alpha}_{\theta} f$ the coordinatewise Caputo fractional gradient of order $\alpha$ anchored at terminal point $c$, $\beta$ the per-coordinate smoothing weights on the next-order $(1+\alpha)$ fractional gradient, $I$ the identity-function normalizer, $M$ the recurrent meta-optimizer with weights $\varphi$ and hidden state $h$, and $j$ the coordinate index.

Reference: Jan Sobotka, Petr Šimánek, Pavel Kordík, "Enhancing Fractional Gradient Descent with Learned Optimizers", arXiv 2025. https://arxiv.org/abs/2510.18783

---
[Back to the Canon](../index.md)

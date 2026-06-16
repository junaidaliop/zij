# λ-FAdaMax

Implements λ-FAdaMax, a fractional-order AdaMax whose second moment decays from the first-order gradient toward the fractional-order gradient.

λ-FAdaMax builds on the AdaMax framework (Adam with an infinity-norm second moment) and the Caputo fractional derivative. The first moment is accumulated from a Caputo fractional-order gradient of order $\alpha$, which encodes gradient history through fractional calculus. The authors observe that the order of the gradient used in the second moment governs the trade-off between convergence speed and precision: a fractional-order second moment raises precision but slows convergence, while a first-order second moment is faster but less precise.

To balance the two, λ-FAdaMax introduces a decay factor $\lambda$ so that the infinity-norm second moment initially relies on the first-order gradient and, as iterations progress, transitions smoothly toward the fractional-order gradient. This yields faster early convergence while retaining the higher final precision of the fractional second moment.

Reference: Guangyao Chen, Zhao Xu, "λ-FAdaMax: A novel fractional-order gradient descent method with decaying second moment for neural network training", Expert Systems with Applications 2025. https://doi.org/10.1016/j.eswa.2025.127156

---
[Back to the Canon](../index.md)

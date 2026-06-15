# TFGD (Tempered)

Implements TFGD (Tempered Fractional Gradient Descent), a fractional-memory optimizer with exponential tempering of stale gradients.

TFGD replaces the plain gradient with a tempered fractional gradient: a weighted sum of all past gradients whose weights are the fractional binomial coefficients of order $\alpha$, additionally damped by an exponential factor $e^{-\lambda j}$ in the lag $j$. The fractional weights inject long-range memory, while the tempering factor $\lambda$ suppresses the contribution of old, noisy gradients so the memory tail decays geometrically rather than algebraically. This stems from the tempered Caputo derivative $D^{\alpha,\lambda}\mathcal{L}(\theta)=\frac{1}{\Gamma(1-\alpha)}\int_0^\infty \tau^{-\alpha} e^{-\lambda\tau}\,\nabla\mathcal{L}(\theta-\tau\delta)\,d\tau$, of which the update below is the discrete analogue.

$$
\begin{aligned}
\theta_{k+1} &= \theta_k - \eta \sum_{j=0}^{k} |w_j|\, e^{-\lambda j}\, \nabla\mathcal{L}(\theta_{k-j}), \\
|w_j| &= \left| \binom{\alpha}{j} \right|, \qquad
\sum_{j=0}^{\infty} |w_j|\, e^{-\lambda j} = (1-e^{-\lambda})^{-\alpha}.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\nabla\mathcal{L}(\theta_{k-j})$ the gradient at lag $j$, $\alpha\in(0,1)$ the fractional order, $\lambda>0$ the tempering parameter, $\binom{\alpha}{j}$ the generalized binomial (fractional difference) coefficient, and $(1-e^{-\lambda})^{-\alpha}=:d_{\alpha,\lambda}$ the alignment coefficient to which the weight sum converges.

Reference: Omar Naifar, "Tempered fractional gradient descent: Theory, algorithms, and robust learning applications", Neural Networks 2025. https://doi.org/10.1016/j.neunet.2025.108005

---
[Back to the Canon](../README.md)

# Quantum Analytic Descent

Implements Quantum Analytic Descent, a hybrid optimizer that fits a classical analytic surrogate of a variational quantum energy landscape and descends on it.

In each outer iteration a small batch of quantum expectation values is measured around the current point $\theta_t$. These fix the coefficients of a closed-form trigonometric model $E(\theta)$ that approximates the true cost surface in a neighborhood. The inner loop then runs ordinary gradient descent on this cheap classical surrogate, whose gradient is available analytically, before re-measuring at the new point. This amortizes expensive quantum queries across many cheap classical steps.

$$
\begin{aligned}
E(\theta) &= A(\theta)\,E^{(A)} + \sum_k \big[ B_k(\theta)\,E_k^{(B)} + C_k(\theta)\,E_k^{(C)} \big] + \sum_{l>k} D_{kl}(\theta)\,E_{kl}^{(D)} \\
a(\phi) &= \tfrac{1+\cos\phi}{2}, \quad b(\phi) = \tfrac{\sin\phi}{2}, \quad c(\phi) = \tfrac{1-\cos\phi}{2} \\
A(\theta) &= \prod_j a(\theta_j), \quad B_k(\theta) = b(\theta_k)\!\!\prod_{j\neq k}\! a(\theta_j), \quad C_k(\theta) = c(\theta_k)\!\!\prod_{j\neq k}\! a(\theta_j) \\
D_{kl}(\theta) &= b(\theta_k)\,b(\theta_l)\!\!\prod_{j\neq k,l}\! a(\theta_j) \\
\theta_{t+1} &= \theta_t - \eta\,\nabla E(\theta_t)
\end{aligned}
$$

where $\theta$ are the variational circuit parameters, $\eta$ the learning rate, and the scalar reference energies $E^{(A)}, E_k^{(B)}, E_k^{(C)}, E_{kl}^{(D)}$ are obtained from quantum measurements at parameter-shifted points (multiples of $\pi/2$) about the current reference, held fixed while the analytic gradient $\nabla E$ drives the inner descent.

Reference: Bálint Koczor, Simon C. Benjamin, "Quantum Analytic Descent", Physical Review Research 2022. https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.023017

---
[Back to the Canon](../README.md)

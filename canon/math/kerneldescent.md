# Kernel Descent

Implements Kernel Descent, an optimizer for variational quantum algorithms that minimizes via reproducing-kernel-Hilbert-space local approximations.

The objective is the expectation $f(\theta) = \langle \psi(\theta) | M | \psi(\theta) \rangle$ of an observable $M$ over the state produced by a parametrized circuit. Such functions lie in a finite-dimensional reproducing kernel Hilbert space $H$ of trigonometric polynomials, with kernel

$$
\tilde{K}(x, z) = \prod_{j=1}^{m} \frac{1 + 2\cos(x_j - z_j)}{3}, \qquad x, z \in \mathbb{R}^m .
$$

At each step, kernel descent evaluates $f$ on a grid around the current point $\theta_t$ and assembles a local surrogate $\tilde{f}_t$ in $H$ that matches $f$ exactly along low-dimensional coordinate subspaces. A hyperparameter $L$ ($1 \le L \le m$) sets the order: the evaluation offsets $q_1, \dots, q_D$ are the distinct points of $\{-\tfrac{2\pi}{3}, 0, \tfrac{2\pi}{3}\}^m$ with at most $L$ nonzero entries. Because that grid makes the kernel Gram matrix the identity, the surrogate has a closed form with no linear solve, and the next iterate comes from classically minimizing it.

$$
\begin{aligned}
\tilde{f}_t(\theta) &= \sum_{j=1}^{D} f(\theta_t + q_j)\, \tilde{K}(q_j,\, \theta - \theta_t), \\
\theta_{t+1} &\in \arg\min_{\theta \in \mathbb{R}^m} \tilde{f}_t(\theta) \quad \text{(one or several classical steps)} .
\end{aligned}
$$

where $\theta$ are the circuit parameters, $\theta_t$ the iterate at step $t$, $f$ the expectation-value objective, $\tilde{K}$ the RKHS kernel, $q_1, \dots, q_D \in \{-\tfrac{2\pi}{3}, 0, \tfrac{2\pi}{3}\}^m$ the parameter-shift offsets with at most $L$ nonzero entries, $D = \sum_{k=0}^{L} 2^k \binom{m}{k}$ the number of circuit evaluations per iteration, and $L$ the approximation order.

Reference: Lars Simon, Holger Eble, Manuel Radons, "Introducing the Kernel Descent Optimizer for Variational Quantum Algorithms", Scientific Reports 2025. https://arxiv.org/abs/2409.10257

---
[Back to the Canon](../README.md)

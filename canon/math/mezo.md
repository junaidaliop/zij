# MeZO

Implements MeZO, a memory-efficient zeroth-order SGD that estimates gradients from forward passes alone.

MeZO fine-tunes language models without backpropagation by replacing the true gradient with the SPSA (Simultaneous Perturbation Stochastic Approximation) estimate: perturb the parameters by $+\epsilon z$ and $-\epsilon z$ along a single random Gaussian direction $z$, evaluate the loss at both points, and use the finite-difference quotient as a scalar coefficient on $z$. This is a rank-1 reconstruction of the gradient that costs only two forward passes per step.

The key trick for memory is in-place perturbation: rather than store $z$, MeZO resets the random number generator with a fixed seed before each perturbation and again before the descent step, regenerating $z$ coordinate-by-coordinate on the fly. The resulting optimizer needs only as much memory as inference, since the only quantity carried between the loss evaluations and the update is the single scalar $\mathrm{projected\_grad}$.

$$
\begin{aligned}
\widehat{\nabla}\mathcal{L}(\theta; B) &= \frac{\mathcal{L}(\theta + \epsilon z; B) - \mathcal{L}(\theta - \epsilon z; B)}{2\epsilon}\, z, \qquad z \sim \mathcal{N}(0, I_d) \\
\theta_{t+1} &= \theta_t - \eta_t\, \widehat{\nabla}\mathcal{L}(\theta_t; B_t)
\end{aligned}
$$

where $\theta \in \mathbb{R}^d$ are the parameters, $\mathcal{L}(\theta; B)$ is the loss on minibatch $B$, $z$ is an i.i.d. standard Gaussian perturbation, $\epsilon$ is the perturbation scale, $\eta_t$ is the learning rate, and the finite-difference quotient $\mathrm{projected\_grad} = (\mathcal{L}(\theta + \epsilon z; B) - \mathcal{L}(\theta - \epsilon z; B))/(2\epsilon)$ is the scalar applied to $z$. The $n$-SPSA variant averages $\widehat{\nabla}\mathcal{L}$ over $n$ sampled directions; $n = 1$ is the default.

Reference: Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, Sanjeev Arora, "Fine-Tuning Language Models with Just Forward Passes", NeurIPS 2023. https://arxiv.org/abs/2305.17333

---
[Back to the Canon](../README.md)

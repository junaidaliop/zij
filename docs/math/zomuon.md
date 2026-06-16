# ZO-Muon

Implements ZO-Muon, a zeroth-order optimizer that orthogonalizes a subspace gradient estimate before the step.

ZO-Muon brings Muon-style orthogonalization to derivative-free training. At each step it draws random perturbation matrices in a low-dimensional subspace, lifts them through a column-orthonormal projection $P$, and forms a randomized finite-difference estimate of the gradient from forward function evaluations only. The matrix sign function (computed by Newton-Schulz iteration) then whitens this estimate by equalizing its active singular directions, and the result is projected back into parameter space for the update.

$$
\begin{aligned}
\hat g_t &= \frac{1}{N_q} \sum_{i=1}^{N_q} \frac{f(\theta_t + \mu P \Psi_i) - f(\theta_t)}{\mu}\, \Psi_i, \\
O_t &= P\, \mathrm{msign}(\hat g_t), \\
\theta_{t+1} &= \theta_t - \gamma_t\, O_t.
\end{aligned}
$$

where $\theta_t$ are the parameters, $\gamma_t$ is the learning rate, $P$ is a column-orthonormal projection matrix (resampled periodically via QR of a random Gaussian matrix), $\Psi_i$ are random Gaussian perturbations sampled in the subspace, $N_q$ is the number of queries per step, $\mu > 0$ is the smoothing radius, and $\mathrm{msign}(\cdot)$ is the matrix sign function approximated by a Newton-Schulz iteration.

Reference: Lang, Wang, Zhang, Hong, Zhang, Yin, and Liu, "Powering Up Zeroth-Order Training via Subspace Gradient Orthogonalization", arXiv preprint 2026. https://arxiv.org/abs/2602.17155

---
[Back to the Canon](../index.md)

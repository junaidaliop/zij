# Rosalin

Implements Rosalin, a shot-frugal optimizer for variational quantum algorithms that couples iCANS adaptive shot allocation with weighted random sampling of Hamiltonian terms.

Rosalin (Random Operator Sampling for Adaptive Learning with Individual Number of shots) targets the measurement budget of estimating gradients on quantum hardware. The cost is a weighted sum of measurable operators $H = \sum_{i=1}^{N} c_i h_i$. Rather than spend an equal number of shots on every term, Rosalin draws each single-shot measurement from a distribution over operators proportional to $|c_i|$ and rescales the outcome to keep the energy estimate unbiased. On top of this it runs the iCANS rule, which sizes the per-parameter shot count $s_\ell$ from a running estimate of the gradient's variance and magnitude, so that noisy or near-flat directions automatically receive more shots while well-resolved ones receive fewer.

Each parameter is updated by parameter-shift gradient descent, where the gradient $g_\ell$ averages $s_\ell$ single-shot parameter-shift estimates and the shot count is set adaptively from exponential moving averages of the gradient variance $\xi_\ell$ and the squared gradient $\chi_\ell^2$:

$$
\begin{aligned}
p_i &= \frac{|c_i|}{M}, \qquad M = \sum_{j=1}^{N} |c_j| \\
g_\ell &= \frac{1}{s_\ell}\sum_{j=1}^{s_\ell} \frac{E_j^{+} - E_j^{-}}{2} \\
s_\ell &\leftarrow \left\lceil \frac{2 L \eta}{2 - L \eta}\cdot \frac{\xi_\ell}{\chi_\ell^{2} + b\,\mu^{k}} \right\rceil \\
\theta_\ell &\leftarrow \theta_\ell - \eta\, g_\ell
\end{aligned}
$$

where $\theta_\ell$ is the $\ell$-th parameter, $\eta$ the learning rate ($0 < \eta < 2/L$), $L$ a Lipschitz constant (taken as $L = M$), and $E_j^{\pm}$ are single-shot estimates of $\langle H\rangle$ at the shifted angles $\theta_\ell \pm \pi/2$, each formed from a sampled operator as $c_i r / p_i$ with measurement outcome $r$. The bias-corrected moments are $\xi_\ell = \xi'_\ell/(1-\mu^{k+1})$ and $\chi_\ell = \chi'_\ell/(1-\mu^{k+1})$ with the running averages $\xi'_\ell \leftarrow \mu\,\xi'_\ell + (1-\mu) S_\ell$ and $\chi'_\ell \leftarrow \mu\,\chi'_\ell + (1-\mu) g_\ell$, where $S_\ell$ is the sample variance of the gradient estimates, $\mu$ the moving-average decay, $b$ a bias regularizer, and $k$ the iteration counter.

Reference: Andrew Arrasmith, Lukasz Cincio, Rolando D. Somma, Patrick J. Coles, "Operator Sampling for Shot-frugal Optimization in Variational Algorithms", arXiv 2020. https://arxiv.org/abs/2004.06252

---
[Back to the Canon](../index.md)

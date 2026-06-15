# Interleaved-ShuffleG

Implements Interleaved-ShuffleG, a differentially private shuffled gradient method that interleaves public and private samples within each epoch.

Each epoch fixes a without-replacement permutation $\pi^{(s)}$ over the data and takes one noisy gradient step per sample. Privacy comes from adding Gaussian noise to every private-sample gradient; convergence is improved by mixing in public samples, whose gradients carry no noise. Of the $n$ steps in an epoch, $n_d$ draw from the private set and the rest from the public set, all under a single interleaved permutation. At each epoch boundary a proximal step applies the regularizer $\psi$.

$$
\begin{aligned}
\theta_{i+1}^{(s)} &= \theta_i^{(s)} - \eta\left(\nabla f\!\left(\theta_i^{(s)}; d_{\pi_i^{(s)}}\right) + \rho_i^{(s)}\right), \quad i = 1,\dots,n,\\
\theta_1^{(s+1)} &= \arg\min_{\theta}\; n\,\psi(\theta) + \frac{1}{2\eta}\left\|\theta - \theta_{n+1}^{(s)}\right\|^2.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $s$ the epoch index, $\pi^{(s)}$ the per-epoch permutation, $d_{\pi_i^{(s)}}$ the $i$-th shuffled sample, $\nabla f$ the per-sample gradient, $\rho_i^{(s)} \sim \mathcal{N}(0, (\sigma^{(s)})^2 I_d)$ the privacy noise (zero for public samples), and $\psi$ the regularizer applied via the proximal step.

Reference: Shuli Jiang, Pranay Sharma, Zhiwei Steven Wu, Gauri Joshi, "Improving the Convergence of Private Shuffled Gradient Methods with Public Data", arXiv 2025. https://arxiv.org/abs/2502.03652

---
[Back to the Canon](../README.md)

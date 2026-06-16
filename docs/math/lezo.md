# LeZO

Implements LeZO, a layer-wise sparse zeroth-order optimizer for memory- and compute-efficient fine-tuning of large language models.

LeZO builds on the MeZO-style SPSA estimator, which approximates the gradient from two forward passes along a single random direction $z$, removing the need to store activations or backpropagate. To cut the perturbation and update cost, LeZO treats whole layers as the unit of sparsity: at each step it randomly keeps a fraction of the layers and zeros the perturbation on the rest, so only the retained parameters are estimated and updated. Because a different layer subset is drawn each step (seeded by $s_t$), full-parameter coverage is still achieved over the course of training.

$$
\begin{aligned}
z'_t &= \mathcal{R}(z_t, \rho, s_t), \quad z_t \sim \mathcal{N}(0, I_d) \\
\hat{g}_t &= \frac{\mathcal{L}(\theta_t + \epsilon z'_t; \mathcal{B}_t) - \mathcal{L}(\theta_t - \epsilon z'_t; \mathcal{B}_t)}{2\epsilon}\, z'_t \\
\theta_{t+1} &= \theta_t - \eta\, \hat{g}_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\mathcal{B}_t$ the minibatch, $\epsilon$ the perturbation scale, and $z_t$ a standard Gaussian direction. The masking operator $\mathcal{R}(z_t, \rho, s_t)$ keeps a $(1-\rho)$ fraction of the layers (selected randomly via seed $s_t$) and sets the perturbation to zero on the sparsified layers, so $\hat{g}_t$ is nonzero only on the retained parameters.

Reference: Fei Wang, Li Shen, Liang Ding, Chao Xue, Ye Liu, Changxing Ding, "Simultaneous Computation and Memory Efficient Zeroth-Order Optimizer for Fine-Tuning Large Language Models", arXiv 2024. https://arxiv.org/abs/2410.09823

---
[Back to the Canon](../index.md)

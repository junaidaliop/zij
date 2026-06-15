# MaZO

Implements MaZO (Masked Zeroth-Order optimization), a sparse masked update for multi-task zeroth-order fine-tuning of large language models.

Zeroth-order methods estimate gradients from finite-difference perturbations and avoid backpropagation, but their high-dimensional perturbation noise makes them unstable, especially under multi-task conflicts. MaZO addresses this by updating only a fixed subset of parameters. It scores every trainable weight with a first- and second-order importance metric, aggregates the scores across tasks, and builds a binary mask $M$ that selects the top-$k$ entries per row. The zeroth-order estimate then drives an SGD update that is applied only on the unfrozen coordinates, shrinking the effective dimension and concentrating the noisy estimate on the parameters that matter.

The gradient is estimated by simultaneous perturbation with a random direction, and the masked update zeroes the step on frozen weights:

$$
\begin{aligned}
\hat{g}_t &= \frac{\mathcal{L}(\theta_t + \epsilon z_t) - \mathcal{L}(\theta_t - \epsilon z_t)}{2\epsilon}\, z_t, \quad z_t \sim \mathcal{N}(0, I_d) \\
\theta_{t+1} &= \theta_t - \eta \,\bigl(\hat{g}_t \odot M\bigr) \\
M_{ij} &= \mathbb{1}\bigl[\, S_{ij} \in \mathrm{top}\text{-}k(S_{i\cdot}) \,\bigr], \quad
S = \sum_{t=1}^{T} \hat{S}^{t}, \quad
S^{t} = S^{t}_{\mathrm{global}} + \alpha\, S^{t}_{\mathrm{greedy}} + \beta\, |W|
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\epsilon$ the perturbation scale, $z_t$ a standard Gaussian direction, $\odot$ element-wise product, $M$ the binary mask selecting the top-$k$ scoring weights in each row, $\hat{S}^{t}$ the row-normalized per-task scores summed over $T$ tasks, $S_{\mathrm{global}}$ a Hessian-based saliency term, $S_{\mathrm{greedy}}$ a per-step loss-reduction term, $|W|$ the weight magnitude, and $\alpha,\beta$ weighting coefficients.

Reference: Zhen Zhang, Yifan Yang, Kai Zhen, Nathan Susanj, Athanasios Mouchtaris, Siegfried Kunzmann, Zheng Zhang, "MaZO: Masked Zeroth-Order Optimization for Multi-Task Fine-Tuning of Large Language Models", 2025. https://arxiv.org/abs/2502.11513

---
[Back to the Canon](../README.md)

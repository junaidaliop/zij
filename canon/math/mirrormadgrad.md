# MirrorMADGRAD

Implements Mirror MADGRAD, the mirror descent form of MADGRAD.


$$
\begin{aligned}
   \lambda_k &= \sqrt[3]{k + 1} \\
   \nu_{k+1} &= \sqrt{\tfrac{k}{k+1}}\, \nu_k + g_k \odot g_k \\
   z_{k+1} &= z_k - \gamma \lambda_k\,
       \frac{g_k}{\sqrt[3]{\nu_{k+1}} + \epsilon} \\
   \theta_{k+1} &= (1 - c)\, \theta_k + c\, z_{k+1}
\end{aligned}
$$

where $\gamma$ is `lr` and $c = 1 - \text{momentum}$. Unlike
MADGRAD this variant updates the dual point $z$ by mirror descent
rather than dual averaging, which is more numerically stable and tends to
work better on large transformer training. It does not support sparse
gradients.

The mirror-descent update shown above follows the facebookresearch/madgrad
implementation; it is not given as an algorithm box in Defazio & Jelassi
(2022), which only states the dual-averaging form.


**Note:** `lr` is not comparable to SGD or Adam and should be set by a sweep.

The mirror variant does not implicitly regularize, so weight decay values
that work with other optimizers are usually appropriate.

Reference: Aaron Defazio and Samy Jelassi, "Adaptivity without Compromise:
A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic
Optimization", Journal of Machine Learning Research, 23(144):1-34, 2022
(preprint arXiv:2101.11075).
https://arxiv.org/abs/2101.11075

---
[Back to the Canon](../README.md)

# SPARTA

Implements SPARTA, an end-to-end differentially private sparse fine-tuning framework that privately selects a trainable subnetwork and then runs sparse DP-SGD on it.

SPARTA (Sparse & PrivAte Row-gradient Thresholding Algorithm) splits a private fine-tuning budget into two phases. In a *mask-selection* phase it estimates which weights matter using clipped, noised per-example gradient magnitudes accumulated over an epoch, groups them by row, and keeps the top-$k$ rows. In a *fine-tuning* phase it runs standard DP-SGD but applies updates only to the selected coordinates through the binary mask $\hat{m}$, so the dense gradient noise is confined to the chosen subnetwork.

Per example $i$ in lot $B_t$, mask selection scores each coordinate by its clipped absolute gradient, sums these over the lot with Gaussian noise, pools the noisy scores into per-row group totals $\tilde{v}_j$, averages over the epoch, and selects the top-$k$ rows; fine-tuning then performs masked DP-SGD with per-example clipping and added noise:

$$
\begin{aligned}
u^{t,i} &= \frac{|g^{t,i}|}{\max\{1,\ \|g^{t,i}\|_2 / C\}} \\
\tilde{u}^t &= \sum_{i \in B_t} u^{t,i} + \mathcal{N}(0, \sigma^2 C^2 I), \qquad
\tilde{v}_j^t = \sum_{i \in \mathcal{G}_j} \tilde{u}_i^t, \qquad
\tilde{v} = \frac{1}{T_b} \sum_{t=1}^{T_b} \tilde{v}^t \\
\hat{z} &= \mathrm{Top\text{-}k}(\tilde{v}), \qquad
\hat{m}_i = \sum_{j \in [q]} \mathbb{I}\big(\hat{z}_j = 1,\ i \in \mathcal{G}_j\big) \\
\tilde{g}^t &= \frac{1}{qn}\Big( \sum_{i \in B_t} \frac{g^{t,i}}{\max\{1,\ \|g^{t,i}\|_2 / C\}} + \mathcal{N}(0, \sigma^2 C^2 I) \Big) \\
\theta^{t+1} &= \theta^t - \eta_t\, \big( \hat{m} \odot \tilde{g}^t \big)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g^{t,i}$ the per-example gradient, $C$ the clipping norm, $\sigma$ the noise multiplier, $\mathcal{G}_j$ the index set of group (row) $j$, $\hat{m}\in\{0,1\}^d$ the selected mask, $q$ the lot sampling count and $n$ the dataset size for the DP-SGD average, $T_b$ the number of selection batches, $\odot$ the Hadamard product, and $\mathcal{N}(0,\sigma^2 C^2 I)$ the Gaussian privacy noise.

Reference: Mehdi Makni, Kayhan Behdin, Gabriel Afriat, Zheng Xu, Sergei Vassilvitskii, Natalia Ponomareva, Hussein Hazimeh, Rahul Mazumder, "SPARTA: An Optimization Framework for Differentially Private Sparse Fine-Tuning", arXiv 2025. https://arxiv.org/abs/2503.12822

---
[Back to the Canon](../README.md)

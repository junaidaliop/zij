# Fromage

Implements Fromage, a per-layer optimizer that scales each gradient by the ratio of the weight norm to the gradient norm.

Fromage (Frobenius matched gradient descent) follows from a bound on the distance between two neural networks: the relative change a layer can tolerate is governed by the relative size of its update. Each layer's gradient is rescaled so the update magnitude is proportional to the layer's own weight norm, making the learning rate $\eta$ control the relative change $\|\Delta W_l\|_F / \|W_l\|_F$ uniformly across all layers. The prefactor $1/\sqrt{1+\eta^2}$ caps the compounding growth that would otherwise arise in scale-invariant layers.

$$
\begin{aligned}
W_l \leftarrow \frac{1}{\sqrt{1+\eta^2}}\left[\, W_l - \eta \, \frac{\|W_l\|_F}{\|g_l\|_F} \, g_l \,\right]
\end{aligned}
$$

where $W_l$ are the parameters of layer $l$, $g_l$ is the gradient of layer $l$, $\|\cdot\|_F$ is the Frobenius norm, and $\eta$ is the learning rate.

Reference: Jeremy Bernstein, Arash Vahdat, Yisong Yue, Ming-Yu Liu, "On the distance between two neural networks and the stability of learning", NeurIPS 2020. https://arxiv.org/abs/2002.03432

---
[Back to the Canon](../index.md)

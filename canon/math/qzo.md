# QZO

Implements QZO, zeroth-order fine-tuning of a quantized network through its continuous quantization scales.

The quantized weights $\bar{\theta}$ stay frozen at their discrete values, so no high-precision weight copy or backpropagation through the quantizer is needed. Instead, QZO treats the per-group quantization scales $\Delta$ as the only trainable parameters and estimates their gradient with a two-point SPSA scheme (Q-SPSA): the scales are perturbed symmetrically by $\pm \epsilon z$ along a random Gaussian direction, and the resulting change in loss yields a scalar directional derivative $d$.

To stabilize updates when the loss landscape over scales is sharp, the directional derivative is clipped to $[-C, C]$ before forming the projected gradient $d\, z$. A ZO-SGD step then descends along this estimate, and a $\max(\cdot, 0)$ projection keeps the scales non-negative.

$$
\begin{aligned}
d &= \frac{\mathcal{L}\big((\Delta + \epsilon z)\odot\bar{\theta};\,\mathcal{B}\big) - \mathcal{L}\big((\Delta - \epsilon z)\odot\bar{\theta};\,\mathcal{B}\big)}{2\epsilon}, \\
d' &= \mathrm{clip}(d,\, -C,\, C), \\
\Delta_{t} &= \max\!\big(\Delta_{t-1} - \gamma\, d'\, z,\; 0\big).
\end{aligned}
$$

where $\bar{\theta}$ are the frozen quantized weights, $\Delta$ are the trainable quantization scales, $z \sim \mathcal{N}(0, I)$ is the random perturbation direction, $\epsilon$ is the perturbation scalar, $\odot$ is the dequantization (scale-times-weight) product, $\mathcal{B}$ is the sampled mini-batch, $C$ is the directional-derivative clipping threshold, and $\gamma$ is the learning rate. The projected gradient estimate is $\hat{\nabla}_{\Delta}\mathcal{L} = d\, z$.

Reference: Sifeng Shang, Jiayi Zhou, Chenyu Lin, Minxian Li, Kaiyang Zhou, "Fine-tuning Quantized Neural Networks with Zeroth-order Optimization", arXiv 2025. https://arxiv.org/abs/2505.13430

---
[Back to the Canon](../README.md)

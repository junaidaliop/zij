# QuZO

Implements QuZO, quantized zeroth-order fine-tuning for low-precision large language models.

QuZO estimates gradients from forward passes alone, avoiding the costly back-propagation and high-precision arithmetic that quantization-aware training normally requires. The classical randomized estimator perturbs the quantized weights by $\pm \epsilon u$ and reads off a finite difference of the loss; QuZO replaces the perturbation directions with stochastically rounded, low-precision vectors so that quantization noise stays unbiased in expectation.

For each of $n$ directions it forms the scalar sensitivity $\mu_i$ from the symmetric loss difference, scales an independently quantized direction $u_{i,2}$ by it, and applies the resulting step under a stochastic-rounding quantizer $Q(\cdot)$, keeping the weights in their quantized format throughout.

$$
\begin{aligned}
\mu_i &= \frac{\mathcal{L}_{\mathcal{B}}(\bar{\theta} + \epsilon\, u_{i,1}) - \mathcal{L}_{\mathcal{B}}(\bar{\theta} - \epsilon\, u_{i,1})}{2\epsilon} \\
\bar{\theta}_{t+1} &= \bar{\theta}_t - \sum_{i=1}^{n} Q\!\left( \frac{\eta_t\, \mu_i}{n}\, u_{i,2} \right)
\end{aligned}
$$

where $\bar{\theta}$ are the quantized parameters, $\eta_t$ the learning rate, $\epsilon$ the perturbation scale, $\mathcal{L}_{\mathcal{B}}$ the minibatch loss, $n$ the number of perturbations, $u_{i,1}, u_{i,2}$ two conditionally independent stochastically quantized perturbation directions, and $Q(\cdot)$ a stochastic-rounding quantizer with $\mathbb{E}[Q(u)] = u$.

Reference: Jiajun Zhou, Yifan Yang, Kai Zhen, Ziyue Liu, Yequan Zhao, Ershad Banijamali, Athanasios Mouchtaris, Ngai Wong, Zheng Zhang, "QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models", arXiv 2025. https://arxiv.org/abs/2502.12346

---
[Back to the Canon](../README.md)

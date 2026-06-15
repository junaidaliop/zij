# M-SAM (Modality-Aware SAM)

Implements M-SAM (Modality-Aware SAM), sharpness-aware minimization restricted to the dominant modality in multimodal learning.

In multimodal training, a single modality often dominates the loss, so applying SAM uniformly across all modalities can over-regularize the weaker ones. M-SAM decomposes the per-batch loss by modality contribution using Shapley values, identifies the dominant modality $m_d$, and applies the SAM ascent perturbation only to that modality's loss $\mathcal{L}_d$. The remaining modalities contribute through ordinary first-order gradients of $\mathcal{L}_s$ evaluated at the unperturbed parameters.

$$
\begin{aligned}
m_d &= \arg\max_{m \in \{1,\dots,M\}} v_m, \quad \mathcal{L}_d(\theta) = v_{m_d}\,\ell, \quad \mathcal{L}_s(\theta) = \sum_{m \neq m_d} v_m\,\ell \\
\epsilon_t &= \rho\,\frac{\nabla \mathcal{L}_d(\theta_{t-1})}{\lVert \nabla \mathcal{L}_d(\theta_{t-1}) \rVert_2} \\
\theta_t &\leftarrow \theta_{t-1} - \eta_t\left[\nabla \mathcal{L}_d(\theta_{t-1} + \epsilon_t) + \nabla \mathcal{L}_s(\theta_{t-1}) + \lambda\,\theta_{t-1}\right]
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $\rho$ the neighborhood radius, $\lambda$ the weight decay, $v_m$ the Shapley contribution of modality $m$ to the loss $\ell = \mathcal{L}(f(x_1^i,\dots,x_M^i;\theta), y^i)$, $\mathcal{L}_d$ the dominant-modality loss, and $\mathcal{L}_s$ the aggregate non-dominant loss.

Reference: Hossein Rajoli, Jie Ji, Xiaolong Ma, Fatemeh Afghah, "Modality-Aware SAM: Sharpness-Aware-Minimization Driven Gradient Modulation for Harmonized Multimodal Learning", arXiv 2025. https://arxiv.org/abs/2510.24919

---
[Back to the Canon](../README.md)

# RLR (Recursive Likelihood Ratio)

Implements RLR (Recursive Likelihood Ratio), a hybrid zeroth/half/first-order estimator for fine-tuning diffusion models from a reward signal.

To differentiate the expected reward $R(x_0)$ through a $T$-step denoising chain, full backpropagation is exact but memory-heavy, while a pure likelihood-ratio (score-function) estimate is memory-light but high-variance. RLR interpolates between them: it picks one short sub-chain of length $h$, propagates a precise gradient through that segment, and replaces every other step with a likelihood-ratio estimate that reads only the reward and the injected noise. This keeps the estimator unbiased while cutting both variance and the activations that must be stored.

For each update, a start index $j \sim \mathcal{U}(1, T-h)$ is drawn and the gradient is assembled from a one-step first-order term, an $h$-length half-order term, and a zeroth-order term over the remaining steps; the resulting estimate is then handed to an outer optimizer (Adam in the paper).

$$
\begin{aligned}
\widehat{\nabla_\theta R}(x_0) &= \underbrace{\frac{\partial \varphi_1(x_1; \theta)}{\partial \theta}^{\!\top} \frac{\mathrm{d} R(x_0)}{\mathrm{d} x_0}}_{\text{one-step first-order}} + \underbrace{R(x_0)\, D_\theta^{\top} \varphi_{j:j+h}(x_{j+h}; \theta)\, \nabla \ln f(z_j)}_{h\text{-length half-order}} + \underbrace{\sum_{i \in C} R(x_0)\, \nabla \ln f(z_i)}_{\text{zeroth-order}} \\
\theta_{t+1} &= \theta_t - \gamma\, \widehat{\nabla_\theta R}(x_0)
\end{aligned}
$$

where $\theta$ are the model parameters, $\gamma$ is the learning rate, $x_t$ is the latent at denoising step $t$ ($x_0$ being the output), $z_i \sim \mathcal{N}(0, \sigma_i^2 I)$ is the noise injected at step $i$ with density $f$, $\varphi_t$ is the one-step denoising transition, $D_\theta \varphi_{j:j+h}$ is the Jacobian of the composed $h$-step transition with respect to $\theta$, $R$ is the reward, $j \sim \mathcal{U}(1, T-h)$ selects the backpropagated sub-chain, and $C = \{1, \dots, T\} \setminus \{j, \dots, j+h\}$ indexes the remaining steps.

Reference: Tao Ren, Zishi Zhang, Jingyang Jiang, Zehao Li, Shentao Qin, Yi Zheng, Guanghao Li, Qianyou Sun, Yan Li, Jiafeng Liang, Xinping Li, Yijie Peng, "Half-order Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer", arXiv 2025. https://arxiv.org/abs/2502.00639

---
[Back to the Canon](../README.md)

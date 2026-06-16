# Sparse MeZO

Implements Sparse MeZO, a memory-efficient zeroth-order optimizer that perturbs and updates only a sparse subset of parameters.

Sparse MeZO extends MeZO by restricting the random perturbation to a mask $m \in \{0,1\}^d$, so the central-difference loss probe and the resulting update touch only the masked coordinates. Following the paper's observation that small-magnitude weights are more important for zeroth-order fine-tuning, the mask selects parameters whose absolute value falls below a per-layer threshold $h$.

As in MeZO, no gradients or activations are stored: the perturbation vector $z \sim \mathcal{N}(0, I_d)$ is drawn from a fixed seed, used to form $\theta \pm \gamma\, m \odot z$ for the two forward passes, and then regenerated from the same seed to apply the scalar-scaled update.

$$
\begin{aligned}
\hat{z}_t &= m \odot z_t, \quad z_t \sim \mathcal{N}(0, I_d) \\
g_t &= \frac{\mathcal{L}(\theta_t + \gamma\, \hat{z}_t) - \mathcal{L}(\theta_t - \gamma\, \hat{z}_t)}{2\gamma} \\
\theta_{t+1} &= \theta_t - \eta\, g_t\, (m \odot z_t)
\end{aligned}
$$

where $\gamma$ is the perturbation scale, $\eta$ is the learning rate, $z_t$ is the Gaussian probe direction (regenerated from a stored seed), $\mathcal{L}$ is the minibatch loss, and the mask entry $m_{i,j} = 1$ when $|\theta_{i,j}| \le h_i$ and $0$ otherwise.

Reference: Yong Liu, Zirui Zhu, Chaoyu Gong, Minhao Cheng, Cho-Jui Hsieh, Yang You, "Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning", ICML 2024. https://arxiv.org/abs/2402.15751

---
[Back to the Canon](../index.md)

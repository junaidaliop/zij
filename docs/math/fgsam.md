# FGSAM

Implements FGSAM (Fast Graph Sharpness-Aware Minimization), a SAM variant for few-shot node classification that replaces SAM's second forward-backward pass through the GNN with a cheap pass through an MLP.

Standard SAM perturbs the weights toward the ascent direction and then evaluates the gradient at the perturbed point, which on a graph requires two full message-passing passes per step. FGSAM keeps the perturbation $\hat\epsilon$ computed from the GNN gradient $g^{\mathrm{gnn}}_t = \nabla_\theta \mathcal{L}_{\mathcal G}(\theta_t; f_{\mathrm{gnn}})$, but evaluates the loss at the perturbed weights through a topology-free MLP $f_{\mathrm{mlp}}$, avoiding the expensive second message-passing pass. To reintroduce the topology signal lost by dropping message-passing, it reuses the already-computed GNN gradient with weight $\lambda$, giving a combined update direction $g^{\mathrm{FGSAM}}_t$.

$$
\begin{aligned}
g^{\mathrm{gnn}}_t &= \nabla_\theta \mathcal{L}_{\mathcal G}(\theta_t; f_{\mathrm{gnn}}) \\
\hat\epsilon_t &= \rho\, \frac{g^{\mathrm{gnn}}_t}{\lVert g^{\mathrm{gnn}}_t \rVert} \\
g^{\mathrm{FGSAM}}_t &= \lambda\, g^{\mathrm{gnn}}_t + \nabla_\theta \mathcal{L}_{\mathbf X}\big(\theta_t + \hat\epsilon_t;\, f_{\mathrm{mlp}}\big) \\
\theta_{t+1} &= \theta_t - \eta\, g^{\mathrm{FGSAM}}_t
\end{aligned}
$$

where $\theta$ are the shared weights, $\eta$ the learning rate, $\rho$ the perturbation radius, $\lambda \ge 0$ the weight on the reused GNN gradient that restores topology information, $\mathcal{L}_{\mathcal G}$ the loss computed with message-passing (GNN $f_{\mathrm{gnn}}$), and $\mathcal{L}_{\mathbf X}$ the loss computed without topology (MLP $f_{\mathrm{mlp}}$). The variant FGSAM+ performs the exact FGSAM step only every $k$ iterations and reuses cached gradient components in between for further speedup.

Reference: Yihong Luo, Yuhan Chen, Siya Qiu, Yiwei Wang, Chen Zhang, Yan Zhou, Xiaochun Cao, Jing Tang, "Fast Graph Sharpness-Aware Minimization for Enhancing and Accelerating Few-Shot Node Classification", NeurIPS 2024. https://arxiv.org/abs/2410.16845

---
[Back to the Canon](../index.md)

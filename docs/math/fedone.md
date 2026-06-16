# FedOne

Implements FedOne, a query-efficient federated black-box discrete prompt learning method that activates a single client per round.

FedOne tackles federated tuning of discrete prompts for a cloud-hosted black-box LLM, where each client can only query the model and never sees its weights or gradients. Each client keeps continuous logits $\alpha$ over a categorical prompt-token distribution, samples prompts via Gumbel-Softmax, and estimates the gradient from loss values alone using a variance-reduced policy-gradient (score-function) estimator. The local logits are then updated by SGD on this black-box estimate.

The central result is that, because the per-round query cost grows with the number of activated clients while the convergence benefit is only sublinear, the optimal choice is to activate exactly one client per aggregation round ($K^* = 1$); the server simply adopts that client's logits as the new global state.

$$
\begin{aligned}
\hat{g}_t &= \frac{1}{I-1}\sum_{r=1}^{I}\big(\ell(\Phi^{r}; \mathcal{B}_t) - \bar{\ell}\big)\,\nabla_{\alpha}\log P(\phi_i^{r}), \quad \bar{\ell} = \frac{1}{I}\sum_{w=1}^{I}\ell(\Phi^{w}; \mathcal{B}_t) \\
\alpha_{t+1} &= \alpha_t - \eta\,\hat{g}_t \\
\alpha &\leftarrow \alpha^{k}, \quad K^* = 1 \;\; \text{(server activates one client per round)}
\end{aligned}
$$

where $\alpha$ are the per-token categorical logits, $\eta$ is the learning rate, $\mathcal{B}_t$ a mini-batch, $\Phi^{r}$ the $r$-th of $I$ prompts sampled from $P = \mathrm{GS}(\alpha)$ (Gumbel-Softmax), $\ell$ the black-box LLM loss, $\bar{\ell}$ the baseline mean loss, $\nabla_{\alpha}\log P(\phi_i^{r})$ the score function, and superscript $k$ indexes the single activated client whose logits become the global $\alpha$.

Reference: Ganyu Wang, Jinjie Fang, Maxwell J. Yin, Bin Gu, Xi Chen, Boyu Wang, Yi Chang, Charles Ling, "FedOne: Query-Efficient Federated Learning for Black-box Discrete Prompt Learning", ICML 2025. https://arxiv.org/abs/2506.14929

---
[Back to the Canon](../index.md)

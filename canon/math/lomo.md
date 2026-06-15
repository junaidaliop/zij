# Lomo

Implements LOMO, low-memory optimization fusing the SGD update into backward.


$$
\theta_{t+1} = \theta_t - \eta\, g_t
$$

The update is applied to each parameter in place as soon as its gradient
is computed during the backward pass, and the gradient is freed
immediately afterward, so the full gradient is never materialized.
Nonzero `weight_decay` $\lambda$ is decoupled, multiplying the
parameter by $1 - \eta\lambda$ before the gradient step. Gradient
norm clipping needs the global norm before any parameter is touched, so
it takes two backward passes: one to gather the norm and one to update.

Reference: Kai Lv, Yuqing Yang, Tengxiao Liu, Qinghui Gao, Qipeng Guo,
Xipeng Qiu, "Full Parameter Fine-tuning for Large Language Models with
Limited Resources", ACL 2024.
https://arxiv.org/abs/2306.09782


**Note:** Drive training with `fused_backward` instead of `loss.backward()` followed by `step()`. When `clip_grad_norm` is set, call `grad_norm` on the loss first. fp16 losses enable a dynamic loss scaler automatically.


---
[Back to the Canon](../README.md)

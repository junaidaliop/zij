# WSM

Implements WSM (Warmup-Stable and Merge), a decay-free learning-rate schedule that replaces explicit decay with checkpoint merging.

During training the learning rate only warms up and then stays constant, so no decay phase or fixed end-step is required. The decay effect is recovered afterward by merging the last $k$ checkpoints into a single averaged model. The paper shows this merge is the dual of an implicit learning-rate decay: averaging recent checkpoints with weights $c_j$ is equivalent to having applied per-step gradient weights $w_i$, so any decay shape (cosine, linear, inverse-sqrt) can be reproduced by choosing the merge coefficients accordingly.

$$
\begin{aligned}
\eta_t &= \begin{cases} \eta\,\dfrac{t}{T_{\mathrm{warmup}}}, & t < T_{\mathrm{warmup}} \\ \eta, & t \ge T_{\mathrm{warmup}} \end{cases} \\
\hat{\theta}_{n+k} &= \sum_{j=0}^{k} c_j\,\theta_{n+j} = \theta_n - \sum_{i=1}^{k} w_i\, g_{n+i-1}, \\
w_i &= \sum_{j=i}^{k} c_j, \qquad c_0 = 1 - w_1,\quad c_j = w_j - w_{j+1},\quad c_k = w_k,
\end{aligned}
$$

where $\theta_{n+j}$ are the saved checkpoints, $\hat{\theta}_{n+k}$ is the merged model, $c_j \ge 0$ with $\sum_{j=0}^k c_j = 1$ are the merge coefficients, $1 \ge w_1 \ge \dots \ge w_k \ge 0$ is the equivalent monotone decay sequence applied to gradients $g$, $\eta$ is the peak learning rate, $T_{\mathrm{warmup}}$ the warmup horizon, and $k$ the number of merged checkpoints.

Reference: Changxin Tian, Jiapeng Wang, Qian Zhao, Kunlong Chen, Jia Liu, Ziqi Liu, Jiaxin Mao, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou, "WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training", arXiv 2025. https://arxiv.org/abs/2507.17634

---
[Back to the Canon](../README.md)

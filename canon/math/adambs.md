# AdamBS

Implements AdamBS, Adam driven by an adaptive bandit sampling distribution over training examples.

Instead of drawing a mini-batch uniformly, AdamBS samples each example $j$ with probability $p_j$ and reweights its gradient by $1/(n p_j)$ to keep the batch estimate unbiased. The sampling distribution is treated as an adversarial multi-armed bandit and updated online with an EXP3-style multiplicative-weights rule whose loss rewards examples whose scaled gradient norm is large, so the sampler concentrates on the most informative points. The reweighted estimate $\hat{G}_t$ then feeds an ordinary Adam step.

At each step a mini-batch $I^t = \{I^t_1,\dots,I^t_K\}$ is drawn from $p^{t-1}$, the unbiased gradient is formed, Adam updates the moments and parameters, and the distribution is updated by reweighting and projecting back onto the floored simplex:

$$
\begin{aligned}
\hat{G}_t &= \frac{1}{K}\sum_{k=1}^{K}\frac{g_{I^t_k}}{n\, p^{t-1}_{I^t_k}} \\
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\,\hat{G}_t,\qquad v_t = \beta_2\, v_{t-1} + (1-\beta_2)\,\hat{G}_t^{\,2} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}},\qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\
\theta_t &= \theta_{t-1} - \alpha_\theta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} \\
l_{t,j} &= \begin{cases} -\dfrac{\lVert g^t_j\rVert^2}{(p^t_j)^2} + \dfrac{L^2}{p_{\min}^2} & j \in I^t \\ 0 & \text{otherwise} \end{cases} \\
\hat{h}_{t,j} &= l_{t,j}\,\frac{\sum_{k=1}^{K}\mathbb{1}(j = I^t_k)}{K\, p^t_j} \\
w^t_j &= p^{t-1}_j\,\exp(-\alpha_p\,\hat{h}_{t,j}) \\
p^t &= \mathrm{arg\,min}_{q \in \mathcal{P}}\; D_{\mathrm{KL}}(q \,\|\, w^t)
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha_\theta$ the learning rate, $g_j$ the gradient on example $j$, $\hat{G}_t$ the importance-weighted batch gradient, $m_t$/$v_t$ the first and second moments with bias-corrected forms $\hat{m}_t$/$\hat{v}_t$, $\beta_1,\beta_2$ the decay rates, $\epsilon$ the stability constant, $n$ the dataset size, $K$ the batch size, $p_j$ the sampling probability of example $j$, $\alpha_p$ the distribution learning rate, $L$ a gradient-norm bound, and $\mathcal{P} = \{p : \sum_j p_j = 1,\ p_j \ge p_{\min}\}$ the simplex with a probability floor onto which $w^t$ is projected in KL divergence.

Reference: Rui Liu, Tianyi Wu, Barzan Mozafari, "Adam with Bandit Sampling for Deep Learning", NeurIPS 2020. https://arxiv.org/abs/2010.12986

---
[Back to the Canon](../README.md)

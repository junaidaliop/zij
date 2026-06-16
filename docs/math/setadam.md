# SET-Adam

Implements SET-Adam, an Adam variant that suppresses the range of adaptive stepsizes through layerwise scaling, $\epsilon$-embedding, and translation.

SET-Adam keeps Adam's moment estimates but reshapes the per-coordinate denominator with three operations (the "SET" of the name). It first down-scales the second moment of each layer by $\cos^2$ of the angle between the layer's moment vector and the all-ones vector, embeds $\epsilon$ under the square root, and then down-translates the resulting denominator by a fraction of its layerwise minimum. Together these shrink the spread of effective stepsizes, which improves generalization over Adam.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\tilde{v}_{l,t} &= v_{l,t} \cdot \frac{\left(\sum_i v_{l,t}[i]\right)^2}{d_l \sum_i v_{l,t}[i]^2} \\
w_{l,t} &= \sqrt{\frac{\tilde{v}_{l,t}}{1-\beta_2^t} + \epsilon} \\
\tilde{w}_{l,t} &= w_{l,t} - \tau \min_{i} w_{l,t}[i] \\
\theta_t &= \theta_{t-1} - \frac{\eta}{1-\beta_1^t} \cdot \frac{m_t}{\tilde{w}_t}
\end{aligned}
$$

where $g_t$ is the gradient, $m_t,v_t$ are the first and second moments with decays $\beta_1,\beta_2$, the index $l$ runs over layers with $d_l$ coordinates each, $\tilde{v}_{l,t}$ is the down-scaled second moment using the squared $\cos$ of the angle to the all-ones vector, $w_{l,t}$ is the $\epsilon$-embedded denominator, $\tilde{w}_{l,t}$ is its down-translation by fraction $\tau=0.5$ of the layerwise minimum, $\eta$ is the learning rate, and divisions are coordinate-wise.

Reference: Guoqiang Zhang, "On Suppressing Range of Adaptive Stepsizes of Adam to Improve Generalisation Performance", arXiv 2023. https://arxiv.org/abs/2302.01029

---
[Back to the Canon](../index.md)

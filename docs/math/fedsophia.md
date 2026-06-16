# Fed-Sophia

Implements Fed-Sophia, a communication-efficient second-order federated optimizer that runs Sophia locally on each client and averages the resulting parameters at the server.

Each client takes Sophia-style steps that precondition the gradient EMA by a diagonal Hessian estimate, element-wise clipped to bound the update. The Hessian diagonal is refreshed only every $\tau$ steps via the Gauss-Newton-Bartlett (GNB) estimator $\mathrm{GNB}(\theta) = B \cdot \hat{g} \odot \hat{g}$, where $\hat{g}$ is the gradient of the loss on labels sampled from the model's softmax over a mini-batch of size $B$; this keeps the per-step cost close to first-order methods. After $J$ local iterations the server averages the client models.

$$
\begin{aligned}
g_t^{(i)} &= \nabla h_i(\theta_t^{(i)}) \\
m_t^{(i)} &= \beta_1 m_{t-1}^{(i)} + (1-\beta_1)\, g_t^{(i)} \\
\hat{v}_t^{(i)} &= \beta_2 \hat{v}_{t-\tau}^{(i)} + (1-\beta_2)\, \mathrm{GNB}(\theta_t^{(i)}) \quad (\text{if } t \bmod \tau = 0,\ \text{else } \hat{v}_t^{(i)} = \hat{v}_{t-1}^{(i)}) \\
\theta_t^{(i)} &\leftarrow \theta_t^{(i)} - \eta\lambda\,\theta_t^{(i)} \\
\theta_{t+1}^{(i)} &= \theta_t^{(i)} - \eta \cdot \mathrm{clip}\!\left(\frac{m_t^{(i)}}{\max(\hat{v}_t^{(i)},\, \epsilon)},\ \rho\right) \\
\Theta_{t+1} &= \frac{1}{N}\sum_{i=1}^{N} \theta_{t+1}^{(i)}
\end{aligned}
$$

where $\theta^{(i)}$ are the parameters on client $i$, $\Theta$ the aggregated server model, $\eta$ the learning rate, $g_t$ the local gradient, $m_t$ the gradient EMA, $\hat{v}_t$ the EMA of the diagonal Hessian estimate, $\beta_1,\beta_2$ the decay rates, $\rho$ the clipping threshold, $\lambda$ the weight decay, $\epsilon$ a stability constant, $\tau$ the Hessian refresh period, $N$ the number of clients, and $\mathrm{clip}(z,\rho)=\max(\min(z,\rho),-\rho)$ applied element-wise.

Reference: Ahmed Elbakary, Chaouki Ben Issaid, Mohammad Shehab, Karim Seddik, Tamer ElBatt, Mehdi Bennis, "Fed-Sophia: A Communication-Efficient Second-Order Federated Learning Algorithm", arXiv 2024. https://arxiv.org/abs/2406.06655

---
[Back to the Canon](../index.md)

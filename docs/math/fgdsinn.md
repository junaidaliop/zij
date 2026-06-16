# FGDSINN

Implements FGDSINN, a Caputo fractional-order gradient learning algorithm for smoothing interval neural networks.

Smoothing interval neural networks (SINNs) represent uncertain inputs as intervals, and training them with the ordinary integer-order gradient tends to be inaccurate and unstable. FGDSINN instead descends along a Caputo fractional-order derivative of the loss, whose non-local memory gives smoother, more accurate parameter updates. To avoid the awkward composite-function derivatives that a full fractional chain rule would require, the method keeps the integer-order chain rule for propagation between layers and applies a simplified Caputo fractional gradient only to the parameters within each layer.

For a weight $w$ at iteration $t$, the simplified Caputo fractional derivative of the loss $E$ is taken about the previous iterate $w_{t-1}$, which contributes the $|w_t - w_{t-1}|^{1-\alpha}$ memory factor and a $\Gamma(2-\alpha)$ normalizer; the parameter then steps against this fractional gradient:

$$
\begin{aligned}
D^{\alpha}_{w} E &= \frac{\partial E}{\partial w_{t-1}} \cdot \frac{1}{\Gamma(2-\alpha)} \cdot |w_t - w_{t-1}|^{\,1-\alpha}, \\
w_{t+1} &= w_t - \eta\, D^{\alpha}_{w} E.
\end{aligned}
$$

where $w$ is a within-layer parameter, $\eta > 0$ is the learning rate, $\alpha \in (0,1)$ is the fractional order, $\Gamma(\cdot)$ is the Gamma function, $\frac{\partial E}{\partial w_{t-1}}$ is the ordinary partial derivative of the loss evaluated through the integer-order chain rule, and $D^{\alpha}_{w} E$ is the simplified Caputo fractional gradient. The factor $|w_t - w_{t-1}|^{1-\alpha}$ encodes the fractional memory between consecutive iterates; $\alpha = 1$ recovers integer-order gradient descent.

Reference: Qiang Shao, Yuanquan Liu, Rui Wang, Yan Liu, "A smoothing interval neural networks-based Caputo fractional-order gradient learning algorithm", International Journal of Machine Learning and Cybernetics 2025. https://doi.org/10.1007/s13042-024-02402-1

---
[Back to the Canon](../index.md)

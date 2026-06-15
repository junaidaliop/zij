# FGD (CNN BP)

Implements FGD (CNN BP), a Caputo fractional-order gradient method for CNN back-propagation with a fixed memory step and an adjustable number of terms.

The method replaces the integer-order weight gradient with a Caputo fractional derivative of order $\alpha$. A naive fractional gradient converges to an extreme point fixed by the derivative definition rather than the true minimizer, so the lower terminal of the Caputo integral is set to the previous iterate $w_{\eta-1}$ (the "fixed memory step"); with $0<\alpha<1$ this recovers convergence to the real extreme point. The infinite Caputo series is truncated to $M$ terms, where larger $M$ approximates the ideal fractional gradient more closely while $M=1$ reduces to keeping only the leading term.

During back-propagation the gradient is transferred across layers in integer order to preserve the chain rule, while the per-layer update gradient is taken in fractional order. For each layer $l$ the weights and biases are updated as

$$
\begin{aligned}
w_{\eta+1}^{[l]} &= w_{\eta}^{[l]} - \mu \sum_{v=1}^{M} \frac{f^{(v)}\!\left(w_{\eta-1}^{[l]}\right)}{\Gamma(v+1-\alpha)} \left(\left|w_{\eta}^{[l]} - w_{\eta-1}^{[l]}\right| + \varphi\right)^{(v-\alpha)} \\
b_{\eta+1}^{[l]} &= b_{\eta}^{[l]} - \mu \sum_{v=1}^{M} \frac{f^{(v)}\!\left(b_{\eta-1}^{[l]}\right)}{\Gamma(v+1-\alpha)} \left(\left|b_{\eta}^{[l]} - b_{\eta-1}^{[l]}\right| + \varphi\right)^{(v-\alpha)}
\end{aligned}
$$

where $w^{[l]}$/$b^{[l]}$ are the layer weights and biases, $\eta$ the iteration index, $\mu$ the learning rate, $\alpha\in(0,1)$ the fractional order, $M$ the number of retained terms, $f^{(v)}(\cdot)$ the $v$-th integer-order derivative of the loss, $\Gamma$ the gamma function, and $\varphi$ a small constant added to avoid non-convergence when $w_\eta = w_{\eta-1}$.

Reference: Mundher Mohammed Taresh, Ningbo Zhu, Talal Ahmed Ali Ali, Mohammed Alghaili, Weihua Guo, "Using a novel fractional-order gradient method for CNN back-propagation", arXiv 2022. https://arxiv.org/abs/2205.00581

---
[Back to the Canon](../README.md)

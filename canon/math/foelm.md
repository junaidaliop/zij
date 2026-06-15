# FOELM

Implements FOELM (Fractional-Order Extreme Learning Machine), an extreme learning machine whose output-layer weights are trained by Caputo fractional-order gradient descent.

In an extreme learning machine the hidden-layer weights are drawn at random and frozen, so only the output-layer weights $\theta$ are learned. FOELM replaces the integer-order gradient of the quadratic error with a Caputo fractional-order gradient, letting the update retain memory of past states through the non-integer order $\alpha$. Each step takes the Caputo derivative of the loss about the previous iterate, which scales the ordinary gradient by $|\theta_t - \theta_{t-1}|^{1-\alpha}/\Gamma(2-\alpha)$, and descends along it.

$$
\begin{aligned}
{}^{C}D^{\alpha} f(\theta) &= \frac{1}{\Gamma(n-\alpha)} \int_{c}^{\theta} \frac{f^{(n)}(\tau)}{(\theta-\tau)^{\alpha-n+1}}\, d\tau, \qquad n-1 < \alpha < n, \\
g_t^{(\alpha)} &= \frac{1}{\Gamma(2-\alpha)}\, \frac{\partial E}{\partial \theta}\Big|_{\theta_t}\, |\theta_t - \theta_{t-1}|^{1-\alpha}, \\
\theta_{t+1} &= \theta_t - \eta\, g_t^{(\alpha)}.
\end{aligned}
$$

where $\theta$ are the output-layer weights, $E$ the quadratic (squared-error) loss, $\eta$ the learning rate, $\alpha \in (0,1)$ the fractional order, $\Gamma$ the Gamma function, ${}^{C}D^{\alpha}$ the Caputo fractional derivative taken about lower terminal $c$ (here the previous iterate $\theta_{t-1}$), and $g_t^{(\alpha)}$ the resulting Caputo fractional gradient.

Reference: Yuanquan Liu, Qiang Shao, Yan Liu, Dakun Yang, "An interval neural network-based Caputo fractional-order extreme learning machine applied to classification", Applied Soft Computing 2024. https://doi.org/10.1016/j.asoc.2024.112310

---
[Back to the Canon](../README.md)

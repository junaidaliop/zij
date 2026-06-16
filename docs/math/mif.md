# MIF

Implements MIF, a mixed integer-fractional gradient descent rule for backpropagation neural networks.

The method fuses the classical integer-order gradient with a Caputo fractional-order gradient of the same loss, so each weight step blends the standard descent direction with a long-memory, nonlocal correction. Within a layer the parameters are updated by this combined rule, while the propagation between layers keeps the integer-order chain rule, avoiding fractional derivatives of composite functions. Because the Caputo derivative of the squared-error in the weight reduces, via the power rule, to a factor $\theta^{1-\alpha}/\Gamma(2-\alpha)$ times the ordinary gradient, the fractional term is a reweighted gradient whose strength is set by the current weight magnitude and the fractional order.

$$
\begin{aligned}
D^{\alpha}_{\theta_t} E &= \frac{\theta_t^{\,1-\alpha}}{\Gamma(2-\alpha)}\, \frac{\partial E}{\partial \theta_t} \\
\theta_{t+1} &= \theta_t - \eta\, \frac{\partial E}{\partial \theta_t} + \tau\, D^{\alpha}_{\theta_t} E
\end{aligned}
$$

where $\theta$ is a layer weight (the paper writes $q_{kj}$ for output-layer and $p_{ji}$ for hidden-layer weights, both updated by the same rule), $E$ is the training error, $\partial E/\partial \theta_t$ is the ordinary integer-order gradient, $D^{\alpha}_{\theta_t} E$ is the Caputo fractional-order gradient reduced through the power rule, $\Gamma$ is the gamma function, $\alpha \in (0,1)$ is the fractional order, $\eta>0$ is the learning rate, and $\tau>0$ weights the fractional contribution.

Reference: Yiqun Zhang, Honglei Xu, Yang Li, Guang Lin, Liyuan Zhang, Chuanjiang Tao, Yonghong Wu, "An Integer-Fractional Gradient Algorithm for Back Propagation Neural Networks", Algorithms 2024. https://doi.org/10.3390/a17050220

---
[Back to the Canon](../index.md)

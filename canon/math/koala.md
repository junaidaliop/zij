# KOALA

Implements KOALA, a Kalman-filter optimizer that treats the parameters as a latent state and the loss as a noisy measurement.

KOALA casts training as state estimation: the weights $\theta$ are the hidden state of a linear dynamical system, and each minibatch loss is a scalar observation driven toward a scheduled target. A predict step inflates the scalar state covariance by the process noise, and an update step applies the Kalman gain, which moves $\theta$ along the gradient by an amount that adapts automatically to the loss gap and the gradient magnitude. Because the gain shrinks as the covariance is consumed, the effective step size is self-regulating and needs no hand-tuned learning rate. The variant shown is KOALA-V (vanilla); KOALA-M augments the state with a velocity component for momentum.

$$
\begin{aligned}
\hat{P}_t &= P_{t-1} + Q \\
K_t &= \frac{\hat{P}_t}{\hat{P}_t\,\lVert g_t \rVert^2 + R} \\
\theta_t &= \theta_{t-1} - K_t \big(L_t(\theta_{t-1}) - L_t^{\mathrm{target}}\big)\, g_t \\
P_t &= \frac{R}{\hat{P}_t\,\lVert g_t \rVert^2 + R}\,\hat{P}_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t = \nabla L_t(\theta_{t-1})$ the minibatch gradient, $P_t$ the (scalar) posterior state covariance, $\hat{P}_t$ its predicted value, $K_t$ the Kalman gain that plays the role of an adaptive step size, $Q$ the process (state) noise, $R$ the measurement noise, $L_t$ the minibatch loss, and $L_t^{\mathrm{target}}$ the scheduled target loss the filter steers toward.

Reference: Aram Davtyan, Sepehr Sameni, Llukman Cerkezi, Givi Meishvilli, Adam Bielski, Paolo Favaro, "KOALA: A Kalman Optimization Algorithm with Loss Adaptivity", arXiv 2021. https://arxiv.org/abs/2107.03331

---
[Back to the Canon](../README.md)

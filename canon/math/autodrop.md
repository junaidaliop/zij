# AutoDrop

Implements AutoDrop, a learning rate scheduler that drops the rate automatically when the angular velocity of the parameter trajectory saturates.

AutoDrop wraps any SGD-style optimizer and decides *when* to cut the learning rate from the geometry of the trajectory rather than from a fixed schedule. At the end of each epoch it forms the step vector $s_i$ between consecutive parameter checkpoints, measures the angle between successive steps, and calls the per-epoch angle the angular velocity $\omega_i$. Early in training the direction keeps turning, so $\omega$ changes; once the optimizer plateaus at the current rate the direction stops turning and $\omega$ flattens.

When the change in angular velocity between two consecutive epochs falls below a threshold $\theta$, the rate is multiplied by a drop factor $\rho \in (0,1)$ (floored at a minimum rate), and $\theta$ is enlarged to tolerate the noisier angles seen at smaller steps.

$$
\begin{aligned}
s_i &= x_{i+1} - x_i \\
\angle(s_i, s_{i-1}) &= \frac{180}{\pi}\,\arccos\!\left(\frac{s_i^{\top} s_{i-1}}{\lVert s_i\rVert\,\lVert s_{i-1}\rVert + \epsilon}\right) \\
\omega_i &= \angle(s_i, s_{i-1}) \\
\text{if } |\omega_t - \omega_{t-1}| &< \theta : \quad
\alpha \leftarrow \max\{\underline{\alpha},\, \rho\,\alpha\}, \quad
\theta \leftarrow \min\{\bar{\theta},\, \tfrac{1}{\rho}\,\theta\}
\end{aligned}
$$

where $x_i$ is the parameter vector at the end of epoch $i$, $s_i$ the per-epoch step, $\omega_i$ the angular velocity (the angle in degrees between successive steps), $\alpha$ the learning rate with lower bound $\underline{\alpha}$, $\rho$ the drop factor, $\theta$ the saturation threshold capped at $\bar{\theta}$, and $\epsilon$ a small stability constant. The drop trigger is checked once the change is averaged over the recent epochs.

Reference: Yunfei Teng, Jing Wang, Anna Choromanska, "AutoDrop: Training Deep Learning Models with Automatic Learning Rate Drop", arXiv 2021. https://arxiv.org/abs/2111.15317

---
[Back to the Canon](../README.md)

# OCAR

Implements OCAR (Online Curvature-Aware Replay), a second-order optimizer for online continual learning that preconditions a replay-augmented gradient with curvature from both new and buffered data.

OCAR treats each step as a constrained natural-gradient update: it descends along the combined gradient of the incoming batch and a replay batch, but moves in the metric induced by the Fisher information of both sources. The buffer Fisher is up-weighted by $1+\lambda$ to penalize directions that would disrupt past tasks, and a Tikhonov term $\tau I$ damps the inverse for stability. The Fisher blocks are estimated with K-FAC and accumulated online via an exponential moving average of the Kronecker factors $A$ (input activations) and $G$ (output gradients), so curvature persists across steps without storing past data densely.

$$
\begin{aligned}
A_{t} &= (1-\beta)\,A_{t-1} + \beta\, a_t a_t^{\top} \\
G_{t} &= (1-\beta)\,G_{t-1} + \beta\, g_t g_t^{\top} \\
F_t &= A_t \otimes G_t \\
\delta_t &= -\,\bigl(F_{N_t} + (1+\lambda)\,F_{B_t} + \tau I\bigr)^{-1}\,\bigl(\nabla_{N_t} + \nabla_{B_t}\bigr) \\
\theta_{t} &= \theta_{t-1} + \gamma\,\delta_t
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ the learning rate, $F_{N_t}$ and $F_{B_t}$ the K-FAC Fisher information matrices on the new-data batch $N_t$ and the replay-buffer batch $B_t$, $\nabla_{N_t}$ and $\nabla_{B_t}$ their respective gradients, $\lambda$ the stability weight on the buffer curvature, $\tau$ the damping term, $\beta$ the EMA rate for the Kronecker factors $A,G$, and $\otimes$ the Kronecker product.

Reference: Edoardo Urettini, Antonio Carta, "Online Curvature-Aware Replay: Leveraging 2nd Order Information for Online Continual Learning", arXiv 2025. https://arxiv.org/abs/2502.01866

---
[Back to the Canon](../index.md)

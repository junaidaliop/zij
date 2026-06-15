# PCGrad

Implements PCGrad, gradient surgery for multi-task learning by projecting conflicting task gradients onto each other's normal plane.

In multi-task optimization, the per-task gradients $g_i$ can point in opposing directions, so that descending on one objective increases another. PCGrad detects a conflict between tasks $i$ and $j$ whenever $g_i \cdot g_j < 0$ (negative cosine similarity) and removes the conflicting component of $g_i$ by projecting it onto the plane normal to $g_j$. Each task gradient is altered in turn against the other tasks in random order; the deconflicted gradients are then summed and handed to a standard optimizer (SGD, Adam, etc.) for the parameter update.

$$
\begin{aligned}
g_i^{\mathrm{PC}} &\leftarrow g_i \\
g_i^{\mathrm{PC}} &\leftarrow g_i^{\mathrm{PC}} - \frac{g_i^{\mathrm{PC}} \cdot g_j}{\lVert g_j \rVert^2}\, g_j \quad \text{if } g_i^{\mathrm{PC}} \cdot g_j < 0, \quad \forall j \neq i \\
\theta_{t+1} &= \theta_t - \eta \sum_i g_i^{\mathrm{PC}}
\end{aligned}
$$

where $g_i = \nabla_\theta \mathcal{L}_i(\theta)$ is the gradient of task $i$, $g_i^{\mathrm{PC}}$ its projected (deconflicted) form, $\theta$ the shared parameters, and $\eta$ the learning rate.

Reference: Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn, "Gradient Surgery for Multi-Task Learning", NeurIPS 2020. https://arxiv.org/abs/2001.06782

---
[Back to the Canon](../README.md)

# SCAFFOLD

Implements SCAFFOLD, federated SGD that corrects client drift with control variates.

In federated optimization, each client takes several local SGD steps before the server averages the results. Because clients have heterogeneous data, their local minima differ, and the averaged model drifts away from the true objective. SCAFFOLD (Stochastic Controlled Averaging) cancels this drift by maintaining a server control variate $c$ and per-client control variates $c_i$. During local updates, each gradient is corrected by $-c_i + c$, which steers every client toward the global descent direction. After local training the control variates are refreshed and aggregated alongside the model.

$$
\begin{aligned}
y_i &\leftarrow y_i - \eta_l\,\bigl(g_i(y_i) - c_i + c\bigr) \\
c_i^{+} &\leftarrow c_i - c + \tfrac{1}{K\eta_l}\,(x - y_i) \\
x &\leftarrow x + \eta_g\,\frac{1}{|S|}\sum_{i\in S}(y_i - x) \\
c &\leftarrow c + \frac{|S|}{N}\sum_{i\in S}(c_i^{+} - c_i)
\end{aligned}
$$

where $x$ is the server model, $y_i$ the local model of client $i$ initialized to $x$, $g_i$ a stochastic gradient on client $i$, $c$ the server control variate, $c_i$ the client control variate, $\eta_l$ and $\eta_g$ the local and global step sizes, $K$ the number of local steps, $S$ the set of sampled clients, and $N$ the total number of clients. The first two lines run $K$ times per round on each client (the control-variate update uses Option II; Option I sets $c_i^{+} \leftarrow g_i(x)$), and the last two are the server aggregation.

Reference: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh, "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020. https://arxiv.org/abs/1910.06378

---
[Back to the Canon](../index.md)

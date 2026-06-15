# Distributed and Communication-Efficient Optimizers

Optimizers in this category target training across many devices or nodes, where memory and inter-worker communication are the main bottlenecks. They shard optimizer state, compress gradient exchange, or synchronize infrequently so that training scales without a proportional increase in bandwidth. Some entries are standalone update rules, while others wrap an inner optimizer with a communication-efficient outer loop.

| Optimizer | Venue | Paper | Code | `zij` |
|---|---|---|---|---|
| [signSGD](math/signsgd-v.md) | ICML 2018 | [signSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434) | [official](https://github.com/jxbz/signSGD) | — |
| [LD-SGD](math/ldsgd.md) | arXiv 2019 | [Communication-Efficient Local Decentralized SGD Methods](https://arxiv.org/abs/1910.09126) | — | — |
| [Local SGD](math/localsgd.md) | ICLR 2019 | [Local SGD Converges Fast and Communicates Little](https://arxiv.org/abs/1805.09767) | [community](https://github.com/epfml/LocalSGD-Code) | — |
| [PowerSGD](math/powersgd.md) | NeurIPS 2019 | [PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization](https://arxiv.org/abs/1905.13727) | — | — |
| [Qsparse-local-SGD](math/qsparselocalsgd.md) | NeurIPS 2019 | [Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification, and Local Computations](https://arxiv.org/abs/1906.02367) | — | — |
| [signProx](math/signprox.md) | ICASSP 2019 | [signProx: One-Bit Proximal Algorithm for Nonconvex Stochastic Optimization](https://arxiv.org/abs/1807.08023) | — | — |
| [APMSqueeze](math/apmsqueeze.md) | arXiv 2020 | [APMSqueeze: A Communication Efficient Adam-Preconditioned Momentum SGD Algorithm](https://arxiv.org/abs/2008.11343) | — | — |
| [DEED-GD](math/deedgd.md) | arXiv 2020 | [DEED: A General Quantization Scheme for Communication Efficiency in Bits](https://arxiv.org/abs/2006.11401) | — | — |
| [FedAC](math/fedac.md) | NeurIPS 2020 | [Federated Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/2006.08950) | — | — |
| [LAGS-SGD](math/lagssgd.md) | ECAI 2020 | [Layer-wise Adaptive Gradient Sparsification for Distributed Deep Learning with Convergence Guarantees](https://arxiv.org/abs/1911.08727) | — | — |
| [rTop-k](math/rtopk.md) | JSAIT 2020 | [rTop-k: A Statistical Estimation Approach to Distributed SGD](https://arxiv.org/abs/2005.10761) | — | — |
| [SCAFFOLD](math/scaffold.md) | ICML 2020 | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378) | — | — |
| [SlowMo](math/slowmo.md) | ICLR 2020 | [SlowMo: Improving Communication-Efficient Distributed SGD with Slow Momentum](https://arxiv.org/abs/1910.00643) | — | — |
| [ZeRO](math/zero.md) | SC 2020 | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | [official](https://github.com/deepspeedai/DeepSpeed) | — |
| [1-bit Adam](math/1bitadam.md) | ICML 2021 | [1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed](https://arxiv.org/abs/2102.02888) | [official](https://github.com/deepspeedai/DeepSpeed) | — |
| [BVR-L-SGD](math/bvrlsgd.md) | ICML 2021 | [Bias-Variance Reduced Local SGD for Less Heterogeneous Federated Learning](https://arxiv.org/abs/2102.03198) | — | — |
| [SQuARM-SGD](math/squarmsgd.md) | JSAIT 2021 | [SQuARM-SGD: Communication-Efficient Momentum SGD for Decentralized Optimization](https://arxiv.org/abs/2005.07041) | — | — |
| [SketchedAMSGrad](math/sketchedamsgrad.md) | ICDM 2022 | [Communication-Efficient Adam-Type Algorithms for Distributed Data Mining](https://arxiv.org/abs/2210.07454) | — | — |
| [0/1 Adam](math/01adam.md) | ICLR 2023 | [Maximizing Communication Efficiency for Large-scale Training via 0/1 Adam](https://arxiv.org/abs/2202.06009) | [official](https://github.com/deepspeedai/DeepSpeed) | — |
| [AdaCGD](math/adacgd.md) | TMLR 2023 | [Adaptive Compression for Communication-Efficient Distributed Training](https://arxiv.org/abs/2211.00188) | — | — |
| [DiLoCo](math/diloco.md) | arXiv 2023 | [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105) | [community](https://github.com/PrimeIntellect-ai/OpenDiloco) | — |
| [Distributed Shampoo](math/distributedshampoo.md) | arXiv 2023 | [A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale](https://arxiv.org/abs/2309.06497) | [official](https://github.com/facebookresearch/optimizers) | — |
| [SPARQ-SGD](math/sparqsgd.md) | TAC 2023 | [SPARQ-SGD: Event-Triggered and Compressed Communication in Decentralized Stochastic Optimization](https://arxiv.org/abs/1910.14280) | — | — |
| [AdaFedAdam](math/adafedadam.md) | TMLCN 2024 | [Accelerating Fair Federated Learning: Adaptive Federated Adam](https://arxiv.org/abs/2301.09357) | [official](https://github.com/li-ju666/adafedadam) | — |
| [DeMo](math/demo.md) | arXiv 2024 | [DeMo: Decoupled Momentum Optimization](https://arxiv.org/abs/2411.19870) | [official](https://github.com/bloc97/DeMo) | — |
| [FADAS](math/fadas.md) | ICML 2024 | [FADAS: Towards Federated Adaptive Asynchronous Optimization](https://arxiv.org/abs/2407.18365) | [official](https://github.com/yujiaw98/FADAS) | — |
| [FAGH](math/fagh.md) | arXiv 2024 | [FAGH: Accelerating Federated Learning with Approximated Global Hessian](https://arxiv.org/abs/2403.11041) | — | — |
| [Fed-Sophia](math/fedsophia.md) | ICC 2024 | [Fed-Sophia: A Communication-Efficient Second-Order Federated Learning Algorithm](https://arxiv.org/abs/2406.06655) | — | — |
| [FedLion](math/fedlion.md) | ICASSP 2024 | [FedLion: Faster Adaptive Federated Optimization with Fewer Communication](https://arxiv.org/abs/2402.09941) | [official](https://github.com/TZW1998/FedLion) | — |
| [FedRepOpt](math/fedrepopt.md) | ACCV 2024 | [FedRepOpt: Gradient Re-parametrized Optimizers in Federated Learning](https://arxiv.org/abs/2409.15898) | [official](https://github.com/StevenLauHKHK/FedRepOpt) | — |
| [FedSTaS](math/fedstas.md) | arXiv 2024 | [FedSTaS: Client Stratification and Client Level Sampling for Efficient Federated Learning](https://arxiv.org/abs/2412.14226) | [official](https://github.com/askjdasf/FedSTaS) | — |
| [FESS-GDA](math/fessgda.md) | AISTATS 2024 | [Stochastic Smoothed Gradient Descent Ascent for Federated Minimax Optimization](https://arxiv.org/abs/2311.00944) | — | — |
| [FLeNS](math/flens.md) | BigData 2024 | [FLeNS: Federated Learning with Enhanced Nesterov-Newton Sketch](https://arxiv.org/abs/2409.15216) | [official](https://github.com/sunnyinAI/FLeNS) | — |
| [MM-PSGD / MC-PSGD](math/mmpsgdmcpsgd.md) | MMAsia-W 2024 | [Distributed Optimization over Block-Cyclic Data](https://arxiv.org/abs/2002.07454) | — | — |
| [OpenDiLoCo](math/opendiloco.md) | arXiv 2024 | [OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training](https://arxiv.org/abs/2407.07852) | [official](https://github.com/PrimeIntellect-ai/OpenDiloco) | — |
| [ADEF](math/adef.md) | arXiv 2025 | [Accelerated Distributed Optimization with Compression and Error Feedback](https://arxiv.org/abs/2503.08427) | — | — |
| [DAT-SGD](math/datsgd.md) | ICML 2025 | [Enhancing Parallelism in Decentralized Stochastic Convex Optimization](https://arxiv.org/abs/2506.00961) | — | — |
| [DeCo-SGD](math/decosgd.md) | arXiv 2025 | [Taming Latency and Bandwidth: A Theoretical Framework and Adaptive Algorithm for Communication-Constrained Training](https://arxiv.org/abs/2507.17346) | — | — |
| [DES-LOC](math/desloc.md) | arXiv 2025 | [DES-LOC: Desynced Low Communication Adaptive Optimizers for Training Foundation Models](https://arxiv.org/abs/2505.22549) | — | — |
| [Dion](math/dion.md) | arXiv 2025 | [Dion: Distributed Orthonormalized Updates](https://arxiv.org/abs/2504.05295) | [official](https://github.com/microsoft/dion) | — |
| [DLAS-R-FTC](math/dlasrftc.md) | CDC 2025 | [Distributed Optimization and Learning for Automated Stepsize Selection with Finite Time Coordination](https://arxiv.org/abs/2508.05887) | — | — |
| [FAdamGC](math/fadamgc.md) | arXiv 2025 | [Gradient Correction in Federated Learning with Adaptive Optimization](https://arxiv.org/abs/2502.02727) | — | — |
| [FedCET](math/fedcet.md) | arXiv 2025 | [Communication Efficient Federated Learning with Linear Convergence on Heterogeneous Data](https://arxiv.org/abs/2503.15804) | — | — |
| [FedIvon](math/fedivon.md) | TMLR 2025 | [Federated Learning with Uncertainty and Personalization via Efficient Second-order Optimization](https://arxiv.org/abs/2411.18385) | — | — |
| [FedMuon](math/fedmuon.md) | arXiv 2025 | [FedMuon: Accelerating Federated Learning with Matrix Orthogonalization](https://arxiv.org/abs/2510.27403) | [official](https://github.com/junkangLiu0/FedMuon) | — |
| [FedOne](math/fedone.md) | ICML 2025 | [FedOne: Query-Efficient Federated Learning for Black-box Discrete Prompt Learning](https://arxiv.org/abs/2506.14929) | — | — |
| [HybridSGD](math/hybridsgd.md) | arXiv 2025 | [Communication-Efficient, 2D Parallel Stochastic Gradient Descent for Distributed-Memory Optimization](https://arxiv.org/abs/2501.07526) | — | — |
| [Kuramoto-FedAvg](math/kuramotofedavg.md) | arXiv 2025 | [Kuramoto-FedAvg: Using Synchronization Dynamics to Improve Federated Learning Optimization under Statistical Heterogeneity](https://arxiv.org/abs/2505.19605) | [official](https://github.com/amuhebwa/Kuramoto-FedAvg) | — |
| [LQ-SGD](math/lqsgd.md) | arXiv 2025 | [Trustworthy Efficient Communication for Distributed Learning using LQ-SGD Algorithm](https://arxiv.org/abs/2506.17974) | — | — |
| [Muon](math/muon.md) | arXiv 2025 | [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) | [official](https://github.com/MoonshotAI/Moonlight) | `Muon` |
| [pFedSOP](math/pfedsop.md) | arXiv 2025 | [pFedSOP: Accelerating Training Of Personalized Federated Learning Using Second-Order Optimization](https://arxiv.org/abs/2506.07159) | — | — |
| [LT-ADMM](math/ltadmm.md) | TAC 2026 | [Communication-Efficient Stochastic Distributed Learning](https://arxiv.org/abs/2501.13516) | — | — |
| [Ringleader ASGD](math/ringleaderasgd.md) | ICLR 2026 | [Ringleader ASGD: The First Asynchronous SGD with Optimal Time Complexity under Data Heterogeneity](https://arxiv.org/abs/2509.22860) | — | — |
| [DECA](math/deca.md) | arXiv 2026 | [DECA: Decentralizing Block-Wise Adam for Efficient LLM Full-Parameter Fine-Tuning on Non-IID Data](https://arxiv.org/abs/2606.03209) | — | — |
| [Ringmaster LMO](math/ringmasterlmo.md) | arXiv 2026 | [Ringmaster LMO: Asynchronous Linear Minimization Oracle Momentum Method](https://arxiv.org/abs/2605.18174) | — | — |
| [SignMuon](math/signmuon.md) | arXiv 2026 | [SignMuon: Communication-Efficient Distributed Muon Optimization](https://arxiv.org/abs/2605.16311) | — | — |
| [Orth-Dion](math/orthdion.md) | arXiv 2026 | [Orth-Dion: Eliminating Geometric Mismatch in Distributed Low-Rank Spectral Optimization](https://arxiv.org/abs/2605.16341) | — | — |
| [EF21-Muon](math/ef21muon.md) | arXiv 2025 | [Error Feedback for Muon and Friends](https://arxiv.org/abs/2510.00643) | — | — |
| [MuonBP](math/muonbp.md) | ICLR 2026 | [MuonBP: Faster Muon via Block-Periodic Orthogonalization](https://arxiv.org/abs/2510.16981) | — | — |
| [CurvaDion](math/curvadion.md) | arXiv 2025 | [CurvaDion: Curvature-Adaptive Distributed Orthonormalization](https://arxiv.org/abs/2512.13728) | — | — |
| [Quasi-Newton FL with Error Feedback](math/quasinewtonflwitherrorfeedback.md) | OPT 2025: Optimization for Machine Learning (NeurIPS 2025 Workshop) | [Quasi-Newton Methods for Federated Learning with Error Feedback](https://opt-ml.org/papers/2025/paper148.pdf) | — | — |
| [DeMuon](math/demuon.md) | arXiv 2025 | [DeMuon: A Decentralized Muon for Matrix Optimization over Graphs](https://arxiv.org/abs/2510.01377) | — | — |
| [HeLoCo](math/heloco.md) | arXiv 2026 | [HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity](https://arxiv.org/abs/2606.00271) | — | — |
| [Decoupled DiLoCo](math/decoupleddiloco.md) | arXiv 2026 | [Decoupled DiLoCo for Resilient Distributed Pre-training](https://arxiv.org/abs/2604.21428) | — | — |
| [Partial Parameter Updates](math/partialparameterupdates.md) | arXiv 2025 | [Partial Parameter Updates for Efficient Distributed Training](https://arxiv.org/abs/2509.22418) | — | — |
| [SparseLoCo](math/sparseloco.md) | arXiv 2025 | [Communication Efficient LLM Pre-training with SparseLoCo](https://arxiv.org/abs/2508.15706) | [official](https://github.com/tplr-ai/SparseLoCo) | — |
| [GASLoC](math/gasloc.md) | arXiv 2026 | [Unifying Local Communications and Local Updates for LLM Pretraining](https://arxiv.org/abs/2606.11081) | — | — |
| [MG-ADSGD](math/mgadsgd.md) | arXiv 2026 | [Accelerated Decentralized Stochastic Gradient Descent for Strongly Convex Optimization](https://arxiv.org/abs/2606.07496) | — | — |
| [Local MixVR](math/localmixvr.md) | arXiv 2026 | [Local MixVR: Breaking the Communication-Sample Dependence in Distributed Learning](https://arxiv.org/abs/2606.01128) | — | — |
| [LOSCAR-SGD](math/loscarsgd.md) | arXiv 2026 | [LOSCAR-SGD: Local SGD with Communication-Computation Overlap and Delay-Corrected Sparse Model Averaging](https://arxiv.org/abs/2605.20866) | — | — |
| [HEW-Local SGD](math/hewlocalsgd.md) | arXiv (math.OC) 2026 | [Heterogeneous-Horizon Exact-Weight Local SGD](https://arxiv.org/abs/2604.24463) | — | — |
| [CAPTAIN (C-ALADIN)](math/captaincaladin.md) | arXiv 2026 | [A Global Convergence Analysis of Consensus ALADIN for Convex Optimization](https://arxiv.org/abs/2606.08112) | — | — |
| [FedPAC](math/fedpac.md) | arXiv 2026 | [Taming Preconditioner Drift: Unlocking the Potential of Second-Order Optimizers for Federated Learning on Non-IID Data](https://arxiv.org/abs/2602.19271) | [official](https://anonymous.4open.science/r/FedPAC-8B24) | — |
| [FedAdamW](math/fedadamw.md) | AAAI 2026 | [FedAdamW: A Communication-Efficient Optimizer with Convergence and Generalization Guarantees for Federated Large Models](https://arxiv.org/abs/2510.27486) | [official](https://github.com/junkangLiu0/FedAdamW) | — |
| [LoRDO](math/lordo.md) | arXiv 2026 | [LoRDO: Distributed Low-Rank Optimization with Infrequent Communication](https://arxiv.org/abs/2602.04396) | — | — |

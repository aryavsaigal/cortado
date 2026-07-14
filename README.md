# cortado

a self learning chess engine

## how it works

![flow chart](./graph.svg)

Training begins by generating approximately **35 million** positions through
incremental self-play. Each outer iteration uses the latest network to evaluate
new positions, producing a continually refreshed dataset without relying on
human games or handcrafted evaluation targets.

The value network represents each position using learned piece, positional, and
side-to-move embeddings. A lightweight transformer encoder models interactions
between all 64 squares before pooling the resulting token representations into a
single board embedding, which is projected by a compact MLP to produce a scalar
evaluation in the range $[-1,1]$.

Instead of supervising every position directly from the final game outcome,
Cortado uses a modified TD($\lambda$) target that gradually shifts supervision
from bootstrapped predictions toward the terminal result as the end of the game
approaches. This avoids the game-length dependence introduced by a naïve linear
ply schedule while still allowing early positions to benefit from bootstrapping.

$$
y_t=\lambda^{N-t}z+\left(1-\lambda^{N-t}\right)\hat V_{\bar\theta}(s_{t+1}),
$$

where $N$ is the game length and $z\in[-1,1]$ is the terminal outcome. We refer
to this as **Horizon-Aware Temporal TD (HAT-TD)**.

Training targets are recomputed before each optimization phase using the latest
network parameters, allowing the value estimates to improve as the model
iteratively refines its own supervision.

# cortado

a self learning chess engine

## how it works

![flow chart](./graph.svg)

We generate approximately **35 million** chess positions incrementally through
self-play, using the current network to evaluate successor positions throughout
training. The value network embeds piece identities, board positions, and
side-to-move, processes the resulting token sequence with a transformer encoder,
pools the token representations into a single board embedding, and projects it
to a scalar evaluation.

Training uses a modified TD($\lambda$) target that applies a
position-dependent weighting,

$$
y_t=\lambda^{N-t}z+\left(1-\lambda^{N-t}\right)\hat V_{\bar\theta}(s_{t+1}),
$$

where $N$ is the game length and $z\in[-1,1]$ is the terminal outcome. We refer
to this as **Horizon-Aware Temporal TD (HAT-TD)**.

Training targets are recomputed before each optimization phase using the latest
network parameters, allowing the value estimates to improve as the model
iteratively refines its own supervision.

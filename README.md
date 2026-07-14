# cortado

a self learning chess engine

## how it works

![flow chart](./graph.svg)

We use a chess engine to generate 35 million positions incrementally through self-play using the network's inference per epoch. The network uses a standard attention head which is first compressed using a Mean Pool/Attention Pool layer and then passed through a compression MLP that converts it to a single floating point number.

The RL loss is computed by modifying the TD-lambda algorithm by utilising a per-position reweighting of the standard $\lambda$-return that avoids the game-length-dependence issue with a naive $ply / length$ linear schedule. We call this the **Horizon-Aware Temporal Return**.

The calibrated value estimate for each board is recomputed each mini-iteration using the same algorithm to fine-tune the score.

### Model

Each board position is represented as a sequence of 64 tokens. The embedding for
square $i$ is

$$
x_i^{(0)}
=
E_{\mathrm{piece}}(p_i)
+
E_{\mathrm{pos}}(i)
+
E_{\mathrm{stm}}(c),
$$

where $p_i$ is the piece occupying square $i$ and $c$ denotes the side to
move.

The embeddings are processed by a transformer block

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right),
$$

followed by residual connections and a feed-forward network. The resulting token
representations are pooled

$$
z
=
\frac{1}{64}\sum_{i=1}^{64}x_i,
$$

(or attention pooling in newer variants) before being projected to a scalar
evaluation

$$
\hat V_\theta(s)=f(z).
$$

### Horizon-Aware TD Target

Rather than using a fixed bootstrap target, each position is assigned a
depth-dependent mixing coefficient

$$
\gamma_t=\lambda^{N-t},
$$

where $N$ is the terminal ply of the game.

Training targets are computed as

$$
y_t
=
\gamma_t z
+
(1-\gamma_t)
\hat V_{\bar\theta}(s_{t+1}),
$$

with $z\in[-1,1]$ denoting the game outcome and
$\hat V_{\bar\theta}$ evaluated without gradient propagation.

The network is optimized using mean squared error

$$
\mathcal L(\theta)
=
\mathbb E
\left[
\left(
\hat V_\theta(s)-y
\right)^2
\right].
$$

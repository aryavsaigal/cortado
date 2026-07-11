import pathlib
import random
import numpy as np
import struct
import torch
import torch.nn as nn

n_dim = 32
path = "./games.bin"


class Data:
    def __init__(self, states, counts, outcome):
        self.states = states
        self.counts = counts
        self.outcome = outcome


def read_data():
    data_list = []
    with open(path, "rb") as f:
        data = f.read(4)
        total_count = struct.unpack("<I", data)[0]
        for _ in range(total_count):
            data = f.read(4)
            state_count = struct.unpack("<I", data)[0]
            data = f.read(1)
            outcome = struct.unpack("<b", data)[0]
            states = []
            for _ in range(state_count):
                data = f.read(64)
                board = list(data)
                data = f.read(1)
                stm = struct.unpack("<B", data)[0]
                states.append((board, stm))
            data_list.append(Data(states, state_count, outcome))

    return data_list


def export_weights(model, path):
    order = []
    with open(path, "wb") as f:
        for name, tensor in model.state_dict().items():
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            if arr.ndim == 2 and "embd" not in name:
                arr = arr.T
            f.write(arr.tobytes())
            order.append(f"{name} {list(arr.shape)}")

    with open(".".join(path.split(".")[:-1] + ["order"]), "w") as f:
        f.write("\n".join(order))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(in_features=n_dim, out_features=n_dim, bias=False)
        self.key = nn.Linear(in_features=n_dim, out_features=n_dim, bias=False)
        self.value = nn.Linear(in_features=n_dim, out_features=n_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_dim, 4 * n_dim), nn.ReLU(), nn.Linear(4 * n_dim, n_dim)
        )

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = q @ k.transpose(-2, -1)
        attn = attn * n_dim**-0.5
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out + x
        out = out + self.mlp(out)

        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_embd = nn.Embedding(num_embeddings=64, embedding_dim=n_dim)
        self.stm_embd = nn.Embedding(num_embeddings=2, embedding_dim=n_dim)
        # self.relative_bias = nn.Embedding(225, n_dim)
        self.head = Head()
        self.piece_embd = nn.Embedding(num_embeddings=13, embedding_dim=n_dim)
        self.compress = nn.Sequential(
            nn.Linear(n_dim, n_dim // 2), nn.ReLU(), nn.Linear(n_dim // 2, 1)
        )

    def forward(self, x, stm):
        embd = (
            self.positional_embd(torch.arange(64, device=device)).unsqueeze(0)
            + self.piece_embd(x)
            + self.stm_embd(stm).unsqueeze(1)
        )
        out = self.head(embd)
        out = self.compress(out.mean(dim=1))
        return out


def preprocess_data(data):
    LAMBDA = 0.8
    num_states = sum(game.counts for game in data)
    print(f"data: {num_states}")
    boards = torch.zeros((num_states, 64), dtype=torch.long).to(device)
    stms = torch.zeros((num_states), dtype=torch.long).to(device)
    result = torch.zeros((num_states), dtype=torch.float32).to(device)

    next_boards, next_stms, next_indices = [], [], []
    i = 0
    for game in data:
        length = game.counts - 1
        outcome = game.outcome
        for ply, (board, stm) in enumerate(game.states):
            boards[i] = torch.tensor(board)
            stms[i] = stm
            if ply == length:
                result[i] = outcome
            else:
                nb, nstm = game.states[ply + 1]
                next_boards.append(nb)
                next_stms.append(nstm)
                next_indices.append((i, LAMBDA ** (length - ply), outcome))
            i += 1

    if next_boards:
        nb_tensor = torch.tensor(next_boards, dtype=torch.long).to(device)
        nstm_tensor = torch.tensor(next_stms, dtype=torch.long).to(device)
        with torch.no_grad():
            model_outs = []
            batch_size = 512
            for start in range(0, len(next_boards), batch_size):
                end = min(start + batch_size, len(next_boards))
                model_outs += (
                    model(nb_tensor[start:end], nstm_tensor[start:end])
                    .to("cpu")
                    .squeeze(-1)
                )

        for j, (i, alpha, outcome) in enumerate(next_indices):
            result[i] = alpha * outcome + (1 - alpha) * model_outs[j]

    return boards, stms, result


device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
print(device)

checkpoint = sorted(
    [p for p in pathlib.Path("./nnue/checkpoints").iterdir() if p.is_file()],
    key=lambda p: int(p.name.split(".")[0].split("-")[1]),
)[-1]


def call_binary(
    count=100,
    depth=2,
    random_moves=6,
    path="./games.bin",
    bin_path="./target/release/cortado",
):
    import subprocess

    seed = random.randint(0, 2**64 - 1)

    result = subprocess.run(
        [bin_path, str(count), str(seed), str(depth), str(random_moves), path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("stderr:", result.stderr)
        raise RuntimeError(f"engine exited with code {result.returncode}")
    print(result.stdout)


model = Model().to(device)
data = read_data()
stats = {}
for game in data:
    stats[game.outcome] = stats.get(game.outcome, 0) + 1
print(stats)
optimiser = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-2,
)

checkpoint = torch.load(checkpoint, weights_only=True)

model.load_state_dict(checkpoint["model_state_dict"])
optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
print(f"Loaded: {checkpoint['iter']}")
max_iter = 5000

for gcount in range(100):
    try:
        print(f"On {gcount} game")
        print("DEBUG: Calling Rust binary")
        call_binary()
        data = read_data()
        stats = {}
        for game in data:
            stats[game.outcome] = stats.get(game.outcome, 0) + 1
        print(stats)
        start = checkpoint["iter"] // max_iter + 1
        for evals in range(start, start + 10):
            boards, stms, result = preprocess_data(data)
            data_size = boards.shape[0]

            def make_batch(size):
                idx = torch.randint(data_size, (size,), device=device)
                return boards[idx], stms[idx], result[idx]

            game_size = 512
            for iter in range(max_iter):
                x_boards, x_stms, ys = make_batch(game_size)

                optimiser.zero_grad()

                pred = model(x_boards, x_stms).squeeze(-1)
                loss = nn.functional.mse_loss(pred, ys)

                loss.backward()
                optimiser.step()
                if iter % 250 == 0:
                    print(f"{iter}: {loss.item()}")
                    print(pred.mean().item(), pred.std().item())
                if iter == max_iter - 1:
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimiser.state_dict(),
                        "loss": loss.item(),
                        "iter": evals * max_iter,
                    }
                    torch.save(checkpoint, f"./nnue/checkpoints/checkpoint-{evals}.pt")

            export_weights(model, "./weights.bin")
    except Exception as e:
        print(f"round: {gcount} failed: {e}")
        continue

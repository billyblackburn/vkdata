"""Unit tests for vktorch (Vulkan GPU autograd, nn layers, CNN, Transformer, & optimizers)."""

import numpy as np
import pytest

import vktorch as vt
from vktorch import F, nn, optim
from vktorch.data import DataLoader, TensorDataset


def _num_grad(fn, x_np: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    grad = np.zeros_like(x_np, dtype=np.float32)
    flat_x = x_np.ravel()
    flat_g = grad.ravel()
    for i in range(flat_x.size):
        old = flat_x[i]
        flat_x[i] = old + eps
        p = fn(x_np)
        flat_x[i] = old - eps
        m = fn(x_np)
        flat_x[i] = old
        flat_g[i] = (p - m) / (2.0 * eps)
    return grad


def test_autograd_finite_differences():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((3, 4)).astype(np.float32) * 0.5
    w_np = rng.standard_normal((4, 3)).astype(np.float32) * 0.5
    b_np = rng.standard_normal((3,)).astype(np.float32) * 0.2

    def eval_fn(x_arr):
        t = vt.Tensor(x_arr, requires_grad=False)
        w = vt.Tensor(w_np, requires_grad=False)
        b = vt.Tensor(b_np, requires_grad=False)
        y = ((t @ w + b).gelu().tanh() + (t @ w).silu()).sum()
        return y.item()

    x = vt.Tensor(x_np.copy(), requires_grad=True)
    w = vt.Tensor(w_np.copy(), requires_grad=True)
    b = vt.Tensor(b_np.copy(), requires_grad=True)
    loss = ((x @ w + b).gelu().tanh() + (x @ w).silu()).sum()
    loss.backward()

    ng_x = _num_grad(eval_fn, x_np.copy())
    np.testing.assert_allclose(x.grad.to_numpy(), ng_x, atol=2e-2, rtol=2e-2)

    # LayerNorm + CrossEntropy gradient check
    ln_x = vt.Tensor(x_np.copy(), requires_grad=True)
    targets = np.array([0, 2, 1], dtype=np.int32)
    logits = F.layer_norm(ln_x) @ vt.Tensor(w_np, requires_grad=False)
    ce = F.cross_entropy(logits, targets)
    ce.backward()

    def eval_ce(x_arr):
        t = vt.Tensor(x_arr, requires_grad=False)
        return F.cross_entropy(F.layer_norm(t) @ vt.Tensor(w_np, requires_grad=False), targets).item()

    ng_ce = _num_grad(eval_ce, x_np.copy())
    np.testing.assert_allclose(ln_x.grad.to_numpy(), ng_ce, atol=2e-2, rtol=2e-2)


def test_mlp_training_and_serialization(tmp_path):
    # Concentric rings classification (nonlinear)
    rng = np.random.default_rng(42)
    n = 400
    r0 = rng.uniform(0.2, 0.8, n // 2)
    r1 = rng.uniform(1.4, 2.1, n // 2)
    th0 = rng.uniform(0, 2 * np.pi, n // 2)
    th1 = rng.uniform(0, 2 * np.pi, n // 2)
    X = np.vstack([
        np.column_stack([r0 * np.cos(th0), r0 * np.sin(th0)]),
        np.column_stack([r1 * np.cos(th1), r1 * np.sin(th1)]),
    ]).astype(np.float32)
    y = np.concatenate([np.zeros(n // 2, dtype=np.int32), np.ones(n // 2, dtype=np.int32)])

    model = nn.Sequential(
        nn.Linear(2, 32, seed=1),
        nn.GELU(),
        nn.LayerNorm(32),
        nn.Linear(32, 32, seed=2),
        nn.ReLU(),
        nn.Linear(32, 2, seed=3),
    )
    opt = optim.AdamW(model.parameters(), lr=3e-2, weight_decay=1e-3)
    sched = optim.CosineAnnealingLR(opt, T_max=25)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True, seed=42)

    losses = []
    for _ in range(25):
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        losses.append(ep_loss / len(loader))

    assert losses[-1] < losses[0] * 0.25
    with vt.no_grad():
        preds = np.argmax(model(vt.Tensor(X)).numpy(), axis=1)
    acc = float(np.mean(preds == y))
    assert acc > 0.95, f"Expected >95% accuracy on concentric rings, got {acc:.2%}"

    # Save & load checkpoint roundtrip
    ckpt = tmp_path / "mlp.npz"
    vt.save(model.state_dict(), ckpt)
    model2 = nn.Sequential(
        nn.Linear(2, 32, seed=99),
        nn.GELU(),
        nn.LayerNorm(32),
        nn.Linear(32, 32, seed=99),
        nn.ReLU(),
        nn.Linear(32, 2, seed=99),
    )
    model2.load_state_dict(vt.load(ckpt))
    with vt.no_grad():
        np.testing.assert_allclose(model(vt.Tensor(X)).numpy(), model2(vt.Tensor(X)).numpy(), atol=1e-5)


def test_cnn_conv2d_and_transformer_training():
    # 1. Tiny CNN (Conv2d + MaxPool2d + Flatten + Linear) on horizontal vs vertical bar images
    rng = np.random.default_rng(7)
    n_img = 64
    imgs = rng.normal(0.0, 0.15, (n_img, 1, 8, 8)).astype(np.float32)
    labels = np.zeros(n_img, dtype=np.int32)
    for i in range(n_img):
        if i % 2 == 0:
            imgs[i, 0, 3:5, :] += 1.5  # horizontal bar
            labels[i] = 0
        else:
            imgs[i, 0, :, 3:5] += 1.5  # vertical bar
            labels[i] = 1

    cnn = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=3, padding=1, seed=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(6 * 4 * 4, 2, seed=2),
    )
    opt_cnn = optim.Adam(cnn.parameters(), lr=2e-2)
    x_t = vt.Tensor(imgs)
    loss0 = F.cross_entropy(cnn(x_t), labels).item()
    for _ in range(18):
        opt_cnn.zero_grad()
        loss = F.cross_entropy(cnn(x_t), labels)
        loss.backward()
        opt_cnn.step()
    assert loss.item() < loss0 * 0.2

    # 2. Tiny Causal Transformer (Embedding + TransformerBlock + LM head)
    class TinyGPT(nn.Module):
        def __init__(self, vocab=16, d_model=32, max_len=8):
            super().__init__()
            self.tok = nn.Embedding(vocab, d_model, seed=1)
            self.pos = nn.Embedding(max_len, d_model, seed=2)
            self.block = nn.TransformerBlock(d_model, num_heads=4, mlp_ratio=2.0, is_causal=True, seed=3)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, seed=4)

        def forward(self, idx):
            B, T = idx.shape
            pos_idx = np.broadcast_to(np.arange(T, dtype=np.int32)[None, :], (B, T))
            h = self.tok(idx) + self.pos(pos_idx)
            h = self.ln(self.block(h))
            return self.head(h)

    gpt = TinyGPT(vocab=12, d_model=32, max_len=8)
    opt_gpt = optim.AdamW(gpt.parameters(), lr=3e-2)
    seq_in = np.array([[0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]], dtype=np.int32)
    seq_tgt = np.array([[1, 2, 3, 4, 5, 6, 7], [5, 4, 3, 2, 1, 0, 1]], dtype=np.int32)

    l0 = F.cross_entropy(gpt(seq_in), seq_tgt).item()
    for _ in range(30):
        opt_gpt.zero_grad()
        l = F.cross_entropy(gpt(seq_in), seq_tgt)
        l.backward()
        opt_gpt.step()
    assert l.item() < l0 * 0.15

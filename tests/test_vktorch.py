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


def test_fused_spirv_kernels_gradients_and_wide_rows():
    rng = np.random.default_rng(123)

    # 1. RMSNorm: finite-difference check for both x and weight
    x_np = rng.standard_normal((3, 5)).astype(np.float32) * 0.8
    w_np = rng.standard_normal((5,)).astype(np.float32) * 0.5 + 1.0
    g_np = rng.standard_normal((3, 5)).astype(np.float32)

    x = vt.Tensor(x_np.copy(), requires_grad=True)
    w = vt.Tensor(w_np.copy(), requires_grad=True)
    y = F.rms_norm(x, w, eps=1e-6)
    (y * vt.Tensor(g_np)).sum().backward()

    ng_x = _num_grad(lambda a: (F.rms_norm(vt.Tensor(a), vt.Tensor(w_np), eps=1e-6) * vt.Tensor(g_np)).sum().item(), x_np.copy())
    ng_w = _num_grad(lambda a: (F.rms_norm(vt.Tensor(x_np), vt.Tensor(a), eps=1e-6) * vt.Tensor(g_np)).sum().item(), w_np.copy())
    np.testing.assert_allclose(x.grad.to_numpy(), ng_x, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(w.grad.to_numpy(), ng_w, atol=2e-2, rtol=2e-2)

    # 2. LayerNorm: finite-difference check for x, weight, and bias
    b_np = rng.standard_normal((5,)).astype(np.float32) * 0.3
    x2 = vt.Tensor(x_np.copy(), requires_grad=True)
    w2 = vt.Tensor(w_np.copy(), requires_grad=True)
    b2 = vt.Tensor(b_np.copy(), requires_grad=True)
    (F.layer_norm(x2, w2, b2, eps=1e-5) * vt.Tensor(g_np)).sum().backward()

    ng_ln_x = _num_grad(lambda a: (F.layer_norm(vt.Tensor(a), vt.Tensor(w_np), vt.Tensor(b_np)) * vt.Tensor(g_np)).sum().item(), x_np.copy())
    ng_ln_w = _num_grad(lambda a: (F.layer_norm(vt.Tensor(x_np), vt.Tensor(a), vt.Tensor(b_np)) * vt.Tensor(g_np)).sum().item(), w_np.copy())
    ng_ln_b = _num_grad(lambda a: (F.layer_norm(vt.Tensor(x_np), vt.Tensor(w_np), vt.Tensor(a)) * vt.Tensor(g_np)).sum().item(), b_np.copy())
    np.testing.assert_allclose(x2.grad.to_numpy(), ng_ln_x, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(w2.grad.to_numpy(), ng_ln_w, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(b2.grad.to_numpy(), ng_ln_b, atol=2e-2, rtol=2e-2)

    # 3. Fused SwiGLU: finite-difference check for gate and up
    gate_np = rng.standard_normal((3, 5)).astype(np.float32)
    up_np = rng.standard_normal((3, 5)).astype(np.float32)
    gate_t = vt.Tensor(gate_np.copy(), requires_grad=True)
    up_t = vt.Tensor(up_np.copy(), requires_grad=True)
    (F.swiglu(gate_t, up_t) * vt.Tensor(g_np)).sum().backward()

    ng_gate = _num_grad(lambda a: (F.swiglu(vt.Tensor(a), vt.Tensor(up_np)) * vt.Tensor(g_np)).sum().item(), gate_np.copy())
    ng_up = _num_grad(lambda a: (F.swiglu(vt.Tensor(gate_np), vt.Tensor(a)) * vt.Tensor(g_np)).sum().item(), up_np.copy())
    np.testing.assert_allclose(gate_t.grad.to_numpy(), ng_gate, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(up_t.grad.to_numpy(), ng_up, atol=2e-2, rtol=2e-2)

    # 4. Fused Softmax: finite-difference check
    sm_x = vt.Tensor(x_np.copy(), requires_grad=True)
    (sm_x.softmax(axis=-1) * vt.Tensor(g_np)).sum().backward()
    ng_sm = _num_grad(lambda a: (vt.Tensor(a).softmax(axis=-1) * vt.Tensor(g_np)).sum().item(), x_np.copy())
    np.testing.assert_allclose(sm_x.grad.to_numpy(), ng_sm, atol=2e-2, rtol=2e-2)

    # 5. Wide rows (D = 384 > 256 workgroup size) vs exact NumPy reference
    xw = rng.standard_normal((4, 384)).astype(np.float32)
    ww = rng.standard_normal((384,)).astype(np.float32)
    bw = rng.standard_normal((384,)).astype(np.float32)
    tgt_w = np.array([17, 299, 383, 0], dtype=np.int32)

    with vt.no_grad():
        # RMSNorm
        ref_rms = (xw / np.sqrt(np.mean(xw * xw, axis=-1, keepdims=True) + 1e-6)) * ww
        np.testing.assert_allclose(F.rms_norm(vt.Tensor(xw), vt.Tensor(ww), eps=1e-6).numpy(), ref_rms, atol=1e-5, rtol=1e-4)

        # LayerNorm
        mu = np.mean(xw, axis=-1, keepdims=True)
        var = np.mean((xw - mu) ** 2, axis=-1, keepdims=True)
        ref_ln = ((xw - mu) / np.sqrt(var + 1e-5)) * ww + bw
        np.testing.assert_allclose(F.layer_norm(vt.Tensor(xw), vt.Tensor(ww), vt.Tensor(bw), eps=1e-5).numpy(), ref_ln, atol=1e-5, rtol=1e-4)

        # Softmax
        ex = np.exp(xw - np.max(xw, axis=-1, keepdims=True))
        ref_sm = ex / np.sum(ex, axis=-1, keepdims=True)
        np.testing.assert_allclose(vt.Tensor(xw).softmax(axis=-1).numpy(), ref_sm, atol=1e-5, rtol=1e-4)

        # CrossEntropy (inference / no_grad path)
        ref_ce = float(-np.mean(np.log(ref_sm[np.arange(4), tgt_w])))
        np.testing.assert_allclose(F.cross_entropy(vt.Tensor(xw), tgt_w).item(), ref_ce, atol=1e-5, rtol=1e-4)

    # 6. LlamaBlock end-to-end training check (RMSNorm + Causal Attention + SwiGLU + CrossEntropy)
    class TinyLlama(nn.Module):
        def __init__(self, vocab=12, d_model=32):
            super().__init__()
            self.tok = nn.Embedding(vocab, d_model, seed=1)
            self.block = nn.LlamaBlock(d_model, num_heads=4, hidden_dim=64, is_causal=True, seed=2)
            self.norm = nn.RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False, seed=3)

        def forward(self, idx):
            return self.head(self.norm(self.block(self.tok(idx))))

    llama = TinyLlama()
    opt_ll = optim.AdamW(llama.parameters(), lr=3e-2)
    seq_in = np.array([[0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]], dtype=np.int32)
    seq_tgt = np.array([[1, 2, 3, 4, 5, 6, 7], [5, 4, 3, 2, 1, 0, 1]], dtype=np.int32)
    l0 = F.cross_entropy(llama(seq_in), seq_tgt).item()
    for _ in range(25):
        opt_ll.zero_grad()
        l = F.cross_entropy(llama(seq_in), seq_tgt)
        l.backward()
        opt_ll.step()
    assert l.item() < l0 * 0.15


def test_spirv_conv2d_and_maxpool2d_gradients():
    rng = np.random.default_rng(77)

    # 1. Conv2d (stride=1 & stride=2, padding=1) finite-difference check for x, weight, bias
    x_np = rng.standard_normal((2, 2, 5, 5)).astype(np.float32) * 0.6
    w_np = rng.standard_normal((3, 2, 3, 3)).astype(np.float32) * 0.5
    b_np = rng.standard_normal((3,)).astype(np.float32) * 0.2

    for stride in (1, 2):
        out_probe = F.conv2d(vt.Tensor(x_np), vt.Tensor(w_np), vt.Tensor(b_np), stride=stride, padding=1)
        g_np = rng.standard_normal(out_probe.shape).astype(np.float32)

        x_t = vt.Tensor(x_np.copy(), requires_grad=True)
        w_t = vt.Tensor(w_np.copy(), requires_grad=True)
        b_t = vt.Tensor(b_np.copy(), requires_grad=True)
        (F.conv2d(x_t, w_t, b_t, stride=stride, padding=1) * vt.Tensor(g_np)).sum().backward()

        ng_x = _num_grad(lambda a: (F.conv2d(vt.Tensor(a), vt.Tensor(w_np), vt.Tensor(b_np), stride=stride, padding=1) * vt.Tensor(g_np)).sum().item(), x_np.copy())
        ng_w = _num_grad(lambda a: (F.conv2d(vt.Tensor(x_np), vt.Tensor(a), vt.Tensor(b_np), stride=stride, padding=1) * vt.Tensor(g_np)).sum().item(), w_np.copy())
        ng_b = _num_grad(lambda a: (F.conv2d(vt.Tensor(x_np), vt.Tensor(w_np), vt.Tensor(a), stride=stride, padding=1) * vt.Tensor(g_np)).sum().item(), b_np.copy())
        np.testing.assert_allclose(x_t.grad.to_numpy(), ng_x, atol=2e-2, rtol=2e-2)
        np.testing.assert_allclose(w_t.grad.to_numpy(), ng_w, atol=2e-2, rtol=2e-2)
        np.testing.assert_allclose(b_t.grad.to_numpy(), ng_b, atol=2e-2, rtol=2e-2)

    # 2. MaxPool2d (both non-overlapping k=2,s=2 and overlapping k=3,s=1)
    # Use distinct values so argmax is unambiguous for finite differences
    xp = (rng.standard_normal((2, 2, 4, 4)).astype(np.float32) * 2.0 + np.arange(64, dtype=np.float32).reshape(2, 2, 4, 4) * 0.07)
    for k, st in ((2, 2), (3, 1)):
        op = F.max_pool2d(vt.Tensor(xp), kernel_size=k, stride=st)
        gp = rng.standard_normal(op.shape).astype(np.float32)
        xt = vt.Tensor(xp.copy(), requires_grad=True)
        (F.max_pool2d(xt, kernel_size=k, stride=st) * vt.Tensor(gp)).sum().backward()
        ng_p = _num_grad(lambda a: (F.max_pool2d(vt.Tensor(a), kernel_size=k, stride=st) * vt.Tensor(gp)).sum().item(), xp.copy(), eps=1e-3)
        np.testing.assert_allclose(xt.grad.to_numpy(), ng_p, atol=2e-2, rtol=2e-2)


def test_rope_kv_cache_and_lora():
    rng = np.random.default_rng(2026)

    # 1. RoPE: finite-difference gradient check + non-zero start_pos equivalence
    x_np = rng.standard_normal((2, 2, 4, 6)).astype(np.float32) * 0.7
    g_np = rng.standard_normal((2, 2, 4, 6)).astype(np.float32)
    x_t = vt.Tensor(x_np.copy(), requires_grad=True)
    y_t = F.apply_rotary_pos_emb(x_t, start_pos=3, theta=10000.0)
    (y_t * vt.Tensor(g_np)).sum().backward()
    ng_rope = _num_grad(
        lambda a: (F.apply_rotary_pos_emb(vt.Tensor(a), start_pos=3, theta=10000.0) * vt.Tensor(g_np)).sum().item(),
        x_np.copy(),
    )
    np.testing.assert_allclose(x_t.grad.to_numpy(), ng_rope, atol=2e-2, rtol=2e-2)

    # 2. KV-Cache + LlamaBlock(use_rope=True): full sequence == prefill + token-by-token decode!
    block = nn.LlamaBlock(embed_dim=16, num_heads=4, hidden_dim=32, is_causal=True, use_rope=True, seed=11)
    block.eval()
    seq_np = rng.standard_normal((2, 7, 16)).astype(np.float32) * 0.5
    with vt.no_grad():
        full_out = block(vt.Tensor(seq_np)).numpy()

        cache = nn.KVCache()
        p0 = block(vt.Tensor(seq_np[:, :4, :]), kv_cache=cache).numpy()
        p1 = block(vt.Tensor(seq_np[:, 4:5, :]), kv_cache=cache).numpy()
        p2 = block(vt.Tensor(seq_np[:, 5:6, :]), kv_cache=cache).numpy()
        p3 = block(vt.Tensor(seq_np[:, 6:7, :]), kv_cache=cache).numpy()
        cached_out = np.concatenate([p0, p1, p2, p3], axis=1)

    assert cache.seq_len == 7
    np.testing.assert_allclose(cached_out, full_out, atol=1e-4, rtol=1e-4)

    # 3. LoRALinear: exact step-0 equivalence, frozen base weights, training, & merge_weights()
    base = nn.Linear(16, 8, seed=5)
    w_orig = base.weight.numpy().copy()
    lora = nn.LoRALinear(base, rank=4, alpha=8.0, seed=6)

    xl = vt.Tensor(rng.standard_normal((12, 16)).astype(np.float32))
    yl = vt.Tensor(rng.standard_normal((12, 8)).astype(np.float32))

    # Step-0 equivalence (lora_B == 0)
    with vt.no_grad():
        np.testing.assert_allclose(lora(xl).numpy(), base(xl).numpy(), atol=1e-6)

    # Fine-tune only LoRA parameters
    trainable = [p for p in lora.parameters() if p.requires_grad]
    assert len(trainable) == 2  # only lora_A and lora_B!
    opt_l = optim.AdamW(trainable, lr=5e-2)
    l0 = F.mse_loss(lora(xl), yl).item()
    for _ in range(30):
        opt_l.zero_grad()
        loss = F.mse_loss(lora(xl), yl)
        loss.backward()
        opt_l.step()
    assert loss.item() < l0 * 0.4
    # Base weights remained frozen!
    np.testing.assert_allclose(base.weight.numpy(), w_orig, atol=0.0)

    # merge_weights() produces exact match with zero LoRA overhead
    merged = lora.merge_weights()
    with vt.no_grad():
        np.testing.assert_allclose(merged(xl).numpy(), lora(xl).numpy(), atol=1e-5)

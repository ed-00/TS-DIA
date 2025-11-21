import torch
from model.linear_attention import causal_linear_attention


def naive_causal(q, k, v, eps=1e-6):
    # very small naive reference implementation: for each step t
    # compute running sums and output accordingly
    b, h, n, d = q.shape
    e = v.shape[-1]
    running_k = torch.zeros((b, h, d), dtype=q.dtype, device=q.device)
    running_kv = torch.zeros((b, h, d, e), dtype=q.dtype, device=q.device)
    outs = []
    for t in range(n):
        q_t = q[..., t, :]
        k_t = k[..., t, :]
        v_t = v[..., t, :]
        running_k = running_k + k_t
        running_kv = running_kv + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        denom = torch.einsum("...d,...d->...", q_t, running_k) + eps
        D_inv = 1.0 / denom
        out_t = torch.einsum("...d,...de->...e", q_t, running_kv) * D_inv.unsqueeze(-1)
        outs.append(out_t.unsqueeze(-2))
    return torch.cat(outs, dim=-2)


def test_causal_small_random():
    torch.manual_seed(0)
    b, h, n, d, e = 2, 3, 8, 16, 5
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, e)

    out_my = causal_linear_attention(q, k, v, chunk_size=4)
    out_ref = naive_causal(q, k, v)

    assert out_my.shape == (b, h, n, e)
    # numerically they should be close
    assert torch.allclose(out_my, out_ref, atol=1e-5, rtol=1e-5)


def test_causal_single_step():
    # Make sure 1-step sequence behaves like non-causal where it's just single step
    torch.manual_seed(1)
    b, h, n, d, e = 1, 1, 1, 8, 3
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, e)

    out = causal_linear_attention(q, k, v, chunk_size=1)
    assert out.shape == (b, h, n, e)


def test_causal_bf16_if_available():
    # Only run bf16 test if CUDA & bf16 are supported on this machine
    if not torch.cuda.is_available():
        return

    # Many GPUs only support bf16 on Ampere+ or with driver support; try to
    # detect bf16 support if present. If not supported, skip.
    try:
        bf16_supported = torch.cuda.is_bf16_supported()
    except AttributeError:
        # Fallback: assume newer CUDA and skip if not sure
        bf16_supported = False

    if not bf16_supported:
        return

    torch.manual_seed(2)
    device = torch.device('cuda')
    b, h, n, d, e = 1, 2, 6, 12, 4
    q = torch.randn(b, h, n, d, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn(b, h, n, e, device=device, dtype=torch.float32)

    # reference computed in float32
    out_ref = naive_causal(q, k, v)

    # run our function in bf16 and cast back to float32 for comparison
    q_bf, k_bf, v_bf = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    out_bf = causal_linear_attention(q_bf, k_bf, v_bf, chunk_size=3)

    # compare in float32 with a relaxed tolerance
    assert out_bf.shape == out_ref.shape
    assert torch.allclose(out_bf.to(torch.float32), out_ref, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    test_causal_small_random()
    test_causal_single_step()
    print('ok')

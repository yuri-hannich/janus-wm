"""Quick sanity check: one batch forward+backward pass with bidirectional training."""

import torch
from module import SIGReg, ARPredictor, MLP, Embedder
from jepa import JEPA
import stable_pretraining as spt


def make_fake_batch(batch_size=4, seq_len=4, img_size=224, action_dim=2, frameskip=5):
    """Create a fake batch mimicking Two-Room data."""
    return {
        "pixels": torch.randn(batch_size, seq_len, 3, img_size, img_size),
        "action": torch.randn(batch_size, seq_len, action_dim * frameskip),
    }


def build_model(embed_dim=192, action_dim=10, history_size=3):
    """Build JanusWM model with bidirectional predictor."""
    encoder = spt.backbone.utils.vit_hf(
        size="tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False
    )
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        bidirectional=True,
    )
    action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    return model


def test_forward_pass():
    print("=" * 60)
    print("JanusWM Bidirectional Training — Sanity Check")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    history_size = 3
    n_preds = 1
    action_dim = 2
    frameskip = 5
    lambd = 0.09

    model = build_model(
        embed_dim=192,
        action_dim=action_dim * frameskip,
        history_size=history_size,
    ).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    dir_params = sum(p.numel() for p in model.predictor.dir_embedding.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Direction embedding params: {dir_params:,} (+{dir_params/total_params*100:.3f}%)")
    print()

    # Create fake batch
    batch = make_fake_batch(batch_size=4, seq_len=history_size + n_preds, action_dim=action_dim, frameskip=frameskip)
    batch = {k: v.to(device) for k, v in batch.items()}

    # --- Forward direction ---
    print("[FWD] Forward prediction...")
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, n_preds:]

    pred_fwd = model.predict(ctx_emb, ctx_act, is_forward=True)
    fwd_loss = (pred_fwd - tgt_emb).pow(2).mean()
    print(f"  pred shape: {pred_fwd.shape}, loss: {fwd_loss.item():.4f}")

    # --- Backward direction ---
    print("[BWD] Backward prediction...")
    # Reverse the sequence
    emb_rev = emb.flip(1)
    act_rev = act_emb.flip(1)

    ctx_emb_rev = emb_rev[:, :history_size]
    ctx_act_rev = act_rev[:, :history_size]
    tgt_emb_rev = emb_rev[:, n_preds:]

    pred_bwd = model.predict(ctx_emb_rev, ctx_act_rev, is_forward=False)
    bwd_loss = (pred_bwd - tgt_emb_rev).pow(2).mean()
    print(f"  pred shape: {pred_bwd.shape}, loss: {bwd_loss.item():.4f}")

    # --- Combined loss ---
    sigreg_loss = sigreg(emb.transpose(0, 1))
    total_loss = 0.5 * (fwd_loss + bwd_loss) + lambd * sigreg_loss
    print(f"\n[TOTAL] fwd={fwd_loss.item():.4f} + bwd={bwd_loss.item():.4f} + "
          f"λ*sigreg={lambd * sigreg_loss.item():.4f} = {total_loss.item():.4f}")

    # --- Backward pass (gradient check) ---
    print("\n[GRAD] Running backward pass...")
    total_loss.backward()

    # Check gradients flow through direction embeddings
    dir_grad = model.predictor.dir_embedding.weight.grad
    print(f"  Direction embedding grad norm: {dir_grad.norm().item():.6f}")
    print(f"  Direction embedding values: FWD={model.predictor.dir_embedding.weight[0, :3].tolist()}")
    print(f"                              BWD={model.predictor.dir_embedding.weight[1, :3].tolist()}")

    # Check encoder gradients too
    enc_grad_norm = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
    print(f"  Encoder total grad norm: {enc_grad_norm:.4f}")

    print("\n" + "=" * 60)
    print("SUCCESS — bidirectional training works end-to-end!")
    print("=" * 60)


if __name__ == "__main__":
    test_forward_pass()

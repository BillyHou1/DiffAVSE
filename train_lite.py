# Ronny (main loop) + Fan (data loading) + Billy (support)
# Training script for LiteAVSEMamba, forward pass, loss, backprop, validate,
# checkpoint.
#
# TODO implement the training loop test
#
# 6 losses, 5 active + 1 disabled: mag 0.9 | phase 0.3 | complex 0.1 |
# consistency 0.1 | SI-SDR 0.3 | time 0.0. You still have to compute
# loss_time = F.l1_loss(clean_audio, audio_g) even though its weight is 0.0,
# the config key lookup crashes if you skip it. Complex and consistency are
# both scaled x2 internally, loss_com = F.mse_loss(...) * 2 etc.
#
# Generator-only training with AdamW, no discriminator. Only include params
# with requires_grad=True cause visual encoder backbone is frozen. Scheduler
# is ExponentialLR with gamma from config.
#
# Validate every N steps, compute PESQ/STOI/SI-SDR on full val set with
# torch.no_grad. Save best model when PESQ beats previous best.
# Checkpoints: g_{step:08d}.pth
#
# NaN safety: check_loss_health to skip bad batches and reload from last
# checkpoint after too many consecutive NaNs, safe_backward with try/except
# around loss.backward(), gradient clipping max_norm=1.0. See utils/util.py.
#
# Dataloader returns 7-tuple clean_audio, clean_mag, clean_pha, clean_com,
# noisy_mag, noisy_pha, video. Move all to GPU with non_blocking=True.

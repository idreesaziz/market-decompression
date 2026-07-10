# Safe Claims — Final Confirmation

## Confirmed θ: `b0c0c6469eccf927`
- θ = [0.24, 0.7, 1.9, -4.28, -4.2, 3.44]
- Markets confirmed (p<0.05): BTCUSDT + ETHUSDT
- Pooled b=5: mean=+0.0627, t=+2.14, p=0.0206
- Effect size: ~+0.0627 log_skill (~6.3% MSE improvement from transfer vs forecast-only)

## Confirmed θ: `71681059323782a0`
- θ = [4.2, -4.5, 3.53, -4.5, 4.2, -4.2]
- Markets confirmed (p<0.05): ETHUSDT
- Pooled b=5: mean=+0.0725, t=+2.14, p=0.0206
- Effect size: ~+0.0725 log_skill (~7.2% MSE improvement from transfer vs forecast-only)

## What Is NOT Claimed
- No universal θ* (SOL does not confirm for either theta)
- No b=10 transfer effect (confirmation was null at b=10)
- The original 4 seed anchors did not replicate independently
- No full-field positivity (only specific θ confirm)

## Boundary Finding
- θ = [4.2, −4.2, 4.2, −4.2, 4.2, −4.2] is confirmed **negative** (pooled t=−2.11)
- Extreme θ-region (alternating large values) is harmful to transfer performance
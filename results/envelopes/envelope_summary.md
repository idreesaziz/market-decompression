# Seed-Phase Envelope Volume Estimates

## Method

- GP: ARD Matern-2.5, noise-aware (per-point alpha = SE^2 / y_std^2)
- Grid: 4096 Sobol points in theta in [-4.5, 4.5]^6
- Posterior samples: N=1000 independent pointwise draws Z_i ~ N(mu_i, sigma_i^2)
- Volume fraction: fraction of grid points where sample > 0 (per draw)
- CI is a conservative lower bound: independent draws ignore positive GP correlations
- Domains: full, core (max|theta|<=3), expanded_shell (3<max|theta|<=4.35),
           extreme_shell (max|theta|>4.35)

## Results

### BTCUSDT

| Budget | Domain | Mean VF | CI90 | CI95 | Analytic (prob>0.5) |
|--------|--------|---------|------|------|---------------------|
| b=5 | full | 0.3705 | [0.3584,0.3823] | [0.3555,0.3855] | 0.0955 |
| b=5 | core | 0.3775 | [0.3362,0.4224] | [0.3276,0.4310] | 0.1006 |
| b=5 | expanded_shell | 0.3695 | [0.3549,0.3836] | [0.3526,0.3866] | 0.0917 |
| b=5 | extreme_shell | 0.3713 | [0.3453,0.3973] | [0.3400,0.4040] | 0.1080 |
| b=10 | full | 0.1955 | [0.1858,0.2056] | [0.1841,0.2073] | 0.0068 |
| b=10 | core | 0.2062 | [0.1724,0.2443] | [0.1667,0.2500] | 0.0115 |
| b=10 | expanded_shell | 0.1951 | [0.1835,0.2065] | [0.1811,0.2088] | 0.0067 |
| b=10 | extreme_shell | 0.1920 | [0.1693,0.2147] | [0.1667,0.2187] | 0.0053 |

### ETHUSDT

| Budget | Domain | Mean VF | CI90 | CI95 | Analytic (prob>0.5) |
|--------|--------|---------|------|------|---------------------|
| b=5 | full | 0.6200 | [0.6091,0.6316] | [0.6069,0.6340] | 0.7512 |
| b=5 | core | 0.6701 | [0.6320,0.7098] | [0.6264,0.7155] | 0.7902 |
| b=5 | expanded_shell | 0.6218 | [0.6091,0.6361] | [0.6057,0.6384] | 0.7548 |
| b=5 | extreme_shell | 0.5893 | [0.5627,0.6133] | [0.5587,0.6187] | 0.7187 |
| b=10 | full | 0.5410 | [0.5283,0.5537] | [0.5261,0.5564] | 0.8057 |
| b=10 | core | 0.5491 | [0.5056,0.5920] | [0.4971,0.6006] | 0.8132 |
| b=10 | expanded_shell | 0.5397 | [0.5254,0.5547] | [0.5217,0.5580] | 0.7989 |
| b=10 | extreme_shell | 0.5425 | [0.5147,0.5693] | [0.5093,0.5720] | 0.8293 |

### SOLUSDT

| Budget | Domain | Mean VF | CI90 | CI95 | Analytic (prob>0.5) |
|--------|--------|---------|------|------|---------------------|
| b=5 | full | 0.3584 | [0.3469,0.3704] | [0.3445,0.3728] | 0.0999 |
| b=5 | core | 0.3465 | [0.3017,0.3851] | [0.2931,0.3937] | 0.0977 |
| b=5 | expanded_shell | 0.3576 | [0.3439,0.3716] | [0.3416,0.3743] | 0.1034 |
| b=5 | extreme_shell | 0.3671 | [0.3400,0.3947] | [0.3346,0.3987] | 0.0867 |
| b=10 | full | 0.2087 | [0.1992,0.2185] | [0.1982,0.2205] | 0.0911 |
| b=10 | core | 0.1869 | [0.1552,0.2184] | [0.1494,0.2270] | 0.0575 |
| b=10 | expanded_shell | 0.2085 | [0.1978,0.2201] | [0.1958,0.2225] | 0.0917 |
| b=10 | extreme_shell | 0.2197 | [0.1973,0.2414] | [0.1933,0.2454] | 0.1040 |

## Cross-Market Envelope at b=5 (full domain)

| Market | Mean VF | CI90 |
|--------|---------|------|
| BTCUSDT | 0.3705 | [0.3584,0.3823] |
| ETHUSDT | 0.6200 | [0.6091,0.6316] |
| SOLUSDT | 0.3584 | [0.3469,0.3704] |

## ETH Envelope: b=5 vs b=10 (full domain)

- b=5: mean=0.6200  CI90=[0.6091,0.6316]
- b=10: mean=0.5410  CI90=[0.5283,0.5537]

---
Note: volume fractions are relative to the sampled domain (uniform Sobol coverage). Not calibrated to absolute theta-space volume. Use for relative comparisons only.
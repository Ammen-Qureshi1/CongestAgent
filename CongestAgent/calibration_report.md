# Simulator vs Mininet (noise-free NSFNET, post warm-up)

Fitted parameters: {'tcp_sigma': 1.4, 'tcp_rho': 0.9, 'udp_mean_on': 4.0, 'm_cubic': 1.1361292008419699, 'm_bbr': 0.8, 'k_cubic_loss': 10.0, 'k_bbr_loss': 1.5, 'transient_gain': 0.5, 'transient_intervals': 4, 'cubic_jitter_k': 0.003, 'bbr_jitter_k': 0.0008}

| load | share congestion (Mininet) | share congestion (sim) | mean util (Mininet) | mean backlog (Mininet) | uncertain share (Mininet) |
|---|---|---|---|---|---|
| 30% | 0.0017 | 0.0002 | 0.079 | 0.009 | 0.015 |
| 60% | 0.0100 | 0.0047 | 0.129 | 0.022 | 0.024 |
| 90% | 0.0518 | 0.0176 | 0.263 | 0.074 | 0.048 |

The simulator has no `uncertain` class and its link utilisation is not read per port, so only the congestion share is fitted. Remaining differences are limits of the flow-level model.
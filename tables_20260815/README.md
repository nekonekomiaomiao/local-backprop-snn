# tables_20260815 batch (2026-08-15)

N=3000 parameter-table measurements, 4-core parallel; results merged into the top-level `mnist_table_results.csv`.
Each subdir contains `run_log.txt` and its own `mnist_table_results.csv`.

| dir | config | frozen acc |
|---|---|---|
| t_a15_k0p2_n3000 | shared seed0 a=1.5e-8 k=0.2 | 0.164 |
| t_a10_k0p2_n3000 | shared seed0 a=1e-8 k=0.2 | 0.156 |
| t_a15_k0_n3000 | shared seed0 a=1.5e-8 k=0 | 0.148 |
| t_a15_k0p2_s1_n3000 | shared seed1 a=1.5e-8 k=0.2 | 0.104 |

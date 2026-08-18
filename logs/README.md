# logs/

On 2026-08-15 (docs cleanup), the run/debug logs previously scattered in the project root were consolidated here.

- `mnist_*.txt/log`: early MNIST training logs
- `table_t*.log(.err)`: 2026-08-13/14 parameter-table batch logs
- `table_a*_n1000.log(.err)`: 2026-08-15 N=1000 parameter-table batch logs
- `eval_ckpts_batch1.log(.err)`: checkpoint batch-evaluation logs
- `profile_*` / `probe_*.log`: early debugging logs

New experiment logs remain in their own experiment dirs (`exp*/.../run_log.txt`, `tables_20260815/.../run_log.txt`, `meanfield_ckpts/.../run_log.txt`) — not in the root.

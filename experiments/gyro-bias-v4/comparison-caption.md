Five primary v4 variants on the same ten local public simulated flights, all
completed. Bars use the unrounded values in `comparison.json`; labels show six
decimal places. Panels report equal-episode mean Sim(3) ATE, median per-episode
rotation RMSE, and maximum fuel over all frames, respectively. Every axis starts
at zero. No confidence intervals or statistical-superiority claim are implied.

The blue current model is the selected fallback under the frozen separate
validation rule. The hatched calibrated-gyro candidate remains an unselected
diagnostic despite its public scores. Public evaluation does not revise the
validation decision. Sim(3) alignment removes global scale; this figure alone
does not establish metric accuracy or physical-robot generalization.

Reproduce with `python experiments/gyro-bias-v4/plot_comparison.py` in an
environment with NumPy and Matplotlib. The script preserves existing outputs;
use `--output` and `--provenance` with new paths for another rendering. Input,
script and image hashes are in `comparison-plot-provenance.json`.

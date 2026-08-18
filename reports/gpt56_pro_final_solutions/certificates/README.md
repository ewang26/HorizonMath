# Fixed benchmark certificates

These files preserve the exact pre-existing certificates evaluated in the
archived GPT-5.6 run. They are retained for reproducibility but are excluded
from model-originated discovery credit.

| Problem | Entries | SHA-256 | Objective |
|---|---:|---|---|
| `autocorr_upper` | 90,000 | `a2d2c953704be161f34a421269464ba9e48ba0fe17a4fd81ff0fd69b26d70d80` | Nonnegative one-sided maximum |
| `autocorr_signed_upper` | 400 | `0e86498ba294fb7a45606e3b8aa62765830fe13ed0539c8e57a8e8b0e49c9fae` | Signed one-sided maximum |

The signed certificate concerns `max(convolve(values, values))`; it should
not be interpreted as a certificate for `max(abs(convolve(values, values)))`.

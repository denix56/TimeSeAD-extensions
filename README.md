# TimeSeAD Extensions

This repository adds optional transform families for the NeutralAD model in `timesead_ext`.

## Optional transform families

NeutralAD can now include three optional transform families in its transform bank:

- **Invertible flows**: stacks of affine coupling layers that learn invertible, channel-wise transformations for time series data. Enable with `use_invertible_transforms=True`. Configure with `invertible_cfg(...)` (e.g., number of flows, hidden size, kernel size).【F:timesead_ext/models/transforms/invertible.py†L1-L118】【F:timesead_ext/models/other/neutral_ad.py†L196-L223】
- **Group transforms**: learnable time-domain warping, shifting, and per-channel scale/bias adjustments to create structured augmentations. Enable with `use_group_transforms=True`. Configure with `group_cfg(...)` (e.g., shift/warp toggles and ranges).【F:timesead_ext/models/transforms/group.py†L1-L146】【F:timesead_ext/models/other/neutral_ad.py†L196-L223】
- **Frequency-orthogonal transforms**: apply orthogonal mixing in the frequency domain (either across channels or frequency blocks) via Cayley-parameterized skew matrices. Enable with `use_freq_ortho_transforms=True`. Configure with `freq_cfg(...)` (e.g., mode, block size).【F:timesead_ext/models/transforms/freq.py†L1-L127】【F:timesead_ext/models/other/neutral_ad.py†L196-L223】

## New configuration flags

NeutralAD accepts new flags for assembling the transform bank:

- `use_invertible_transforms`: include the invertible flow family.
- `use_group_transforms`: include the group transform family.
- `use_freq_ortho_transforms`: include the frequency-orthogonal family.
- `keep_base_transforms`: keep the original `SeqTransformNet` transforms (default `True`). Set to `False` if you want only the optional families.
- `transform_families`: provide additional custom transform families (list of lists) to append after the built-ins.

These flags can be combined, and at least one transform must be configured.【F:timesead_ext/models/other/neutral_ad.py†L196-L229】

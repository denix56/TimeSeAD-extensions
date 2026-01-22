# TimeSeAD Extensions

This repository contains extension models and utilities for the TimeSeAD library.

## NeutralAD encoder configuration

`NeutralAD` supports multiple encoder backends via the `encoder_type` argument. The encoder is configured with
`encoder_cfg`, which is passed directly to the selected encoder class.

### `encoder_type="patchtst"`

Uses `PatchTSTEncoder`, which tokenizes each channel into patches before running a transformer stack.

Example:

```python
from timesead_ext.models.other import NeutralAD

model = NeutralAD(
    ts_channels=3,
    seq_len=96,
    encoder_type="patchtst",
    encoder_cfg={
        "patch_len": 16,
        "patch_stride": 8,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
    },
    pooling="mean",
    proj_head="mlp",
    proj_cfg={"hidden_dim": 256, "dropout": 0.1},
)
```

### `encoder_type="itransformer"`

Uses `ITransformerEncoder`, which projects each variate (channel) into the model dimension and applies transformer
blocks across variates.

Example:

```python
from timesead_ext.models.other import NeutralAD

model = NeutralAD(
    ts_channels=5,
    seq_len=64,
    encoder_type="itransformer",
    encoder_cfg={
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
    },
    pooling="meanmax",
    proj_head="linear",
)
```

## Pooling and projection heads

`NeutralAD` combines encoder outputs with a pooling strategy and an optional projection head:

- `pooling`
  - `"base"`: use the encoder's internal pooling (configure via `encoder_cfg["pooling"]`).
  - `"mean"`: mean pooling over tokens.
  - `"meanmax"`: concatenated mean and max pooling.
  - `"attn"`: attention pooling.
- `proj_head`
  - `"base"` or `"linear"`: linear projection into the latent dimension.
  - `"mlp"`: MLP projection head, configure with `proj_cfg` (`hidden_dim`, `dropout`).
  - `"identity"`/`"none"`: skip projection.

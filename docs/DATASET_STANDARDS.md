# Dataset Standards (Landmarks-based)

We train primarily on **landmark sequences** rather than raw pixels.

## Why landmarks?
- Smaller data → faster training
- Better privacy
- More robust to background/lighting

## Folder layout

```
data/
  .gitkeep
  raw/
    videos/                # optional (if you record video)
  processed/
    samples/               # per-sample landmark sequences + label
    splits/                # train/val/test lists
```

## One training sample

Each sample is a **sequence** of frames with a single label.

- Sequence length: variable at capture time
- During training: padded/truncated to fixed `T` (e.g., 30 frames)

### Recommended storage format (v1)

Per sample file: `data/processed/samples/<label>/<sample_id>.npz`

`npz` fields:
- `X`: float32 array shaped `(T, F)` where:
  - `T` = fixed window length (e.g., 30)
  - `F` = feature dimension per frame (hands + pose + visibility flags)
- `y`: int64 scalar class id
- `label`: string label name (for debugging)

## Landmark feature definition (v1)

We build one feature vector per frame by concatenating:

### Hands (MediaPipe Hands)

For each hand we have 21 landmarks with `(x, y, z)`.

Per hand we store:
- `present` (1 float): 1.0 if detected else 0.0
- `handedness` (1 float): +1.0 for Right, -1.0 for Left, 0.0 if missing
- `landmarks` (21 × 3 floats): `(x, y, z)`

So per hand: `1 + 1 + 63 = 65` floats.

Two hands: `130` floats.

### Pose (MediaPipe Pose)

Pose has 33 landmarks with `(x, y, z, visibility)`.

We store:
- `pose_present` (1 float)
- `pose_landmarks` (33 × 4 floats)

So pose: `1 + 132 = 133` floats.

### Total feature dimension (F)

`F = 130 (hands) + 133 (pose) = 263` floats per frame.

Source of truth for `F` and feature ordering: `backend/app/landmarks.py`.

## Normalization (high level)

We will normalize landmarks so the model generalizes across camera distance:

- translate: center around a reference point (e.g., pose mid-hip or wrist)
- scale: divide by a body scale estimate (e.g., shoulder width)
- optional: mirror left-handed samples to right-handed canonical space (v1 option)

✅ For v1 we enable canonical mirroring: if only the left hand is detected in a frame, we mirror X and place it into the right-hand slot.

We will implement this in `tools/` and keep it identical for training and inference.

## Class design

We include:
- `NO_SIGN` (blank)

Plus your chosen vocabulary classes.

## Capture protocol (high-level)

- Capture each sign multiple times per person.
- Vary distance, angle, speed.
- Include negative examples (NO_SIGN) between signs.

Next: we’ll write a concrete capture protocol and scripts in `tools/`.


# Data Collection Protocol (v1)

Goal: collect consistent samples for a 50-word vocabulary using **landmarks**.

## What we collect per sample

Each sample becomes a single `.npz` file containing:
- `X`: `(T, F)` landmark features (padded/truncated)
- `y`: integer class id
- `label`: string label

We do **not** need to store the raw face/video for training, but you may keep optional videos for auditing.

## Recommended capture settings

- Camera: laptop webcam is okay
- Resolution: 640×480 (capture), downsample during streaming
- Lighting: bright, stable lighting
- Background: plain wall if possible
- Framing: upper body visible (hands + torso)

## Dynamic gesture window

We use a fixed training window `T`:
- v1 default: `T = 30` frames
- capture FPS: ~15 fps
- window duration: ~2 seconds

## Sample counts (minimum viable)

For early v1 training (small but workable):

- People: 5–10 people (more is better)
- Per class per person: 30–50 samples

For 50 classes:
- 5 people × 30 samples × 50 classes = 7,500 samples

Also collect **NO_SIGN**:
- 2–3× more NO_SIGN than any single class (because in real life “nothing” happens most of the time)

## Train/Val/Test split (important)

To avoid overestimating accuracy, split by **person**, not by sample.

Example (10 people):
- Train: 7 people
- Val: 2 people
- Test: 1 person

This tests generalization to new users.

## Capture procedure per class

For each word label:

1) Start in neutral position (hands down)
2) Wait 0.5s
3) Perform the sign naturally
4) Hold final pose briefly (0.2–0.5s)
5) Return to neutral

We will later implement automatic segmentation with motion energy; for v1 we keep it simple.

## Quality checklist

Reject a sample if:
- hands not detected for most frames (for hand-dominant signs)
- body not visible for pose-dependent signs
- sign performed too fast/too slow compared to typical examples
- heavy occlusion (hands behind body)

## Handedness handling (v1)

We have two strategies:

1) **Mirror to canonical** (recommended):
   - map left-handed samples into a “right-handed” coordinate system
   - reduces data needs

2) **Keep as-is**:
   - model learns both but needs more data

We will pick one after your confirmation.

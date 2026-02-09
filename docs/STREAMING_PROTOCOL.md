# Streaming Protocol (WebSocket)

We use a **two-message** pattern per frame for simplicity and performance:

1) Client sends **JSON text** `frame_meta`
2) Client sends **binary** payload (JPEG or WebP bytes)

This avoids base64 overhead while keeping the protocol beginner-friendly.

## Client → Server

### 1) `frame_meta` (text message)

```json
{
  "type": "frame_meta",
  "frame_id": "uuid-or-incrementing-string",
  "ts_ms": 1730000000000,
  "encoding": "jpeg",
  "width": 320,
  "height": 240
}
```

### 2) Binary frame payload (bytes message)

- raw bytes of encoded image (JPEG/WebP)
- payload corresponds to the **most recent** `frame_meta`

## Server → Client

### `prediction` (text message)

```json
{
  "type": "prediction",
  "frame_id": "...",
  "token": "NO_SIGN",
  "confidence": 0.99,
  "latency_ms": 12,
  "debug": {
    "hands_detected": 0,
    "pose_detected": false
  }
}
```

Notes:
- During early development we will send a prediction for every received frame.
- Later we will send both **live** (unstable) and **committed** tokens after smoothing.

# Frontend (React 18 + Tailwind)

## Run

```bash
cd frontend
npm install
npm run dev
```

Default backend WebSocket URL is `ws://localhost:8000/ws`.

Set a custom URL by either:

1. Defining `VITE_WS_URL` in your environment, or
2. Editing it in the UI input at runtime.

## Included

- WebRTC webcam capture (`getUserMedia`)
- Frame sampling + JPEG encoding
- WebSocket streaming to backend
- Live/unstable and committed prediction views
- Debounce stabilization UI and sentence builder
- Confidence tier colors and performance metrics
- Live log viewer (`Check Logs`) for WebSocket/prediction events
- Uploaded video testing panel using `/predict/video`
- Theme toggle, responsive layout, and ARIA labels

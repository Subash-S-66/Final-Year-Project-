import { useCallback, useEffect, useRef, useState } from "react";

export function useWebcamCapture({
  width = 640,
  height = 480,
  facingMode = "user",
  jpegQuality = 0.7,
} = {}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraState, setCameraState] = useState("idle");
  const [error, setError] = useState("");

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState("stopped");
  }, []);

  const startCamera = useCallback(async () => {
    setError("");
    setCameraState("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: width },
          height: { ideal: height },
          facingMode,
        },
        audio: false,
      });

      streamRef.current = stream;

      if (!videoRef.current) {
        throw new Error("Video element is not ready");
      }

      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraState("active");
    } catch (cameraError) {
      setCameraState("error");
      setError(cameraError instanceof Error ? cameraError.message : "Could not access webcam");
    }
  }, [facingMode, height, width]);

  const captureFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return null;
    }

    const outWidth = video.videoWidth || width;
    const outHeight = video.videoHeight || height;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }

    canvasRef.current.width = outWidth;
    canvasRef.current.height = outHeight;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) {
      return null;
    }

    ctx.drawImage(video, 0, 0, outWidth, outHeight);

    return new Promise((resolve) => {
      canvasRef.current.toBlob(
        (blob) => {
          resolve(blob ?? null);
        },
        "image/jpeg",
        jpegQuality,
      );
    });
  }, [height, jpegQuality, width]);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  return {
    videoRef,
    cameraState,
    error,
    startCamera,
    stopCamera,
    captureFrame,
  };
}

import { useCallback, useEffect, useRef, useState } from "react";

const WS_STATE = {
  idle: "idle",
  connecting: "connecting",
  connected: "connected",
  disconnected: "disconnected",
  error: "error",
};

export function useWebsocket({
  url,
  onMessage,
  autoReconnect = true,
  reconnectDelayMs = 1500,
}) {
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const shouldReconnectRef = useRef(autoReconnect);
  const onMessageRef = useRef(onMessage);
  const [status, setStatus] = useState(WS_STATE.idle);

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    clearReconnectTimer();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus(WS_STATE.disconnected);
  }, [clearReconnectTimer]);

  const connect = useCallback(() => {
    if (!url) {
      setStatus(WS_STATE.error);
      return;
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    clearReconnectTimer();
    shouldReconnectRef.current = autoReconnect;
    setStatus(WS_STATE.connecting);

    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus(WS_STATE.connected);
    };

    ws.onmessage = (event) => {
      let payload = event.data;
      if (typeof payload === "string") {
        try {
          payload = JSON.parse(payload);
        } catch (error) {
          payload = { type: "text", raw: event.data, parseError: String(error) };
        }
      }
      onMessageRef.current?.(payload);
    };

    ws.onerror = () => {
      setStatus(WS_STATE.error);
    };

    ws.onclose = () => {
      wsRef.current = null;
      setStatus(WS_STATE.disconnected);

      if (shouldReconnectRef.current) {
        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, reconnectDelayMs);
      }
    };
  }, [autoReconnect, clearReconnectTimer, reconnectDelayMs, url]);

  const sendBinary = useCallback((bytes) => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    socket.send(bytes);
    return true;
  }, []);

  const sendJson = useCallback((value) => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    socket.send(JSON.stringify(value));
    return true;
  }, []);

  useEffect(() => {
    return () => {
      shouldReconnectRef.current = false;
      clearReconnectTimer();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [clearReconnectTimer]);

  return {
    status,
    isConnected: status === WS_STATE.connected,
    connect,
    disconnect,
    sendBinary,
    sendJson,
  };
}

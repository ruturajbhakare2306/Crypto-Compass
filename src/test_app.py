"""Live Binance OBI collection and rolling prediction service."""

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone

import websocket

from func import Rolling_prediction


URL = "wss://stream.binance.com:443/ws/btcusdt@depth10@100ms"
OBI_WINDOW = 12


class LivePredictionService:
    """Keep the latest live prediction and OBI window in thread-safe state."""

    def __init__(self):
        self.predictor = Rolling_prediction()
        self.lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.websocket_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.websocket = None
        self.obi_values = deque(maxlen=OBI_WINDOW)
        self.prediction_history = deque(maxlen=5)
        self.prediction = None
        self.probabilities = None
        self.error = None
        self.connection_status = "Not started"
        self.messages_received = 0
        self.last_message_at = None

    def start(self, *, reset=False):
        """Start collecting live data, optionally beginning a fresh prediction window."""
        with self.lifecycle_lock:
            if self.thread is not None and self.thread.is_alive():
                return

            if reset:
                with self.lock:
                    self.predictor.window.clear()
                    self.obi_values.clear()
                    self.prediction_history.clear()
                    self.prediction = None
                    self.probabilities = None
                    self.error = None
                    self.messages_received = 0
                    self.last_message_at = None

            self.connection_status = "Connecting"
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _on_message(self, ws, message):
        try:
            if self.stop_event.is_set():
                return

            with self.lock:
                self.messages_received += 1
                self.last_message_at = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )

            data = json.loads(message)
            bids = data["bids"]
            asks = data["asks"]
            if not bids or not asks:
                return

            bid_quantity = sum(float(level[1]) for level in bids) / len(bids)
            ask_quantity = sum(float(level[1]) for level in asks) / len(asks)
            total_quantity = bid_quantity + ask_quantity
            obi = (
                (bid_quantity - ask_quantity) / total_quantity
                if total_quantity
                else 0.0
            )

            with self.lock:
                if self.stop_event.is_set():
                    return

                result = self.predictor.predict(obi)
                if self.stop_event.is_set():
                    return

                self.obi_values.append(obi)
                self.error = None
                if result is not None:
                    prediction, probabilities = result
                    self.prediction = prediction[-1]
                    self.probabilities = [float(p) for p in probabilities[-1]]
                    self.prediction_history.append(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(
                                timespec="milliseconds"
                            ),
                            "prediction": self.prediction,
                            "probabilities": list(self.probabilities),
                        }
                    )
        except Exception as exc:
            with self.lock:
                self.error = f"Message processing error: {exc}"

    def _on_error(self, ws, error):
        with self.lock:
            self.connection_status = "Connection error"
            self.error = str(error)

    def _on_close(self, ws, close_status_code, close_msg):
        with self.lock:
            if not self.stop_event.is_set():
                self.connection_status = "Disconnected"
                detail = f"WebSocket closed ({close_status_code}): {close_msg}"
                self.error = detail

    def _on_open(self, ws):
        if self.stop_event.is_set():
            ws.close()
            return

        with self.lock:
            if self.stop_event.is_set():
                ws.close()
            else:
                self.connection_status = "Connected"
                self.error = None

    def _run(self):
        while not self.stop_event.is_set():
            with self.websocket_lock:
                if self.stop_event.is_set():
                    return

                current_websocket = websocket.WebSocketApp(
                    URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.websocket = current_websocket

            if self.stop_event.is_set():
                current_websocket.close()
                return

            current_websocket.run_forever()

            with self.websocket_lock:
                if self.websocket is current_websocket:
                    self.websocket = None

            if not self.stop_event.is_set():
                self.stop_event.wait(2)

    def snapshot(self):
        """Return a consistent copy of the latest values for the UI."""
        with self.lock:
            return {
                "running": (
                    self.thread is not None
                    and self.thread.is_alive()
                    and not self.stop_event.is_set()
                ),
                "prediction": self.prediction,
                "probabilities": list(self.probabilities)
                if self.probabilities is not None
                else None,
                "obi_values": list(self.obi_values),
                "prediction_history": [
                    {
                        **record,
                        "probabilities": list(record["probabilities"]),
                    }
                    for record in self.prediction_history
                ],
                "error": self.error,
                "connection_status": self.connection_status,
                "messages_received": self.messages_received,
                "last_message_at": self.last_message_at,
                "obi_count": len(self.obi_values),
            }

    def stop(self):
        """Stop the feed and wait for any in-flight prediction to finish."""
        with self.lifecycle_lock:
            self.stop_event.set()
            with self.lock:
                self.connection_status = "Stopped"
            with self.websocket_lock:
                websocket = self.websocket
                self.websocket = None
            thread = self.thread

            close_error = None
            if websocket is not None:
                try:
                    websocket.close()
                except Exception as exc:
                    close_error = exc

            if thread is not None and thread is not threading.current_thread():
                thread.join()

            if close_error is not None:
                raise RuntimeError("Could not close the live prediction feed.") from close_error


def main():
    """Optional console runner for the same live service."""
    service = LivePredictionService()
    service.start()
    try:
        while True:
            time.sleep(0.25)
            snapshot = service.snapshot()
            if snapshot["prediction"] is not None:
                print(
                    snapshot["prediction"],
                    ":",
                    snapshot["probabilities"],
                )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        service.stop()


if __name__ == "__main__":
    main()

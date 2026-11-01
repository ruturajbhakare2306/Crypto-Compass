"""Live order-book OBI collection and rolling prediction service.

The default source is Binance BTCUSDT. Custom WebSocket sources are supported
when each message contains a complete order-book snapshot with `bids` and
`asks` arrays in [price, quantity] form.
"""

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlparse

import websocket

from func import Rolling_prediction


DEFAULT_URL = "wss://stream.binance.com:443/ws/btcusdt@depth10@100ms"
OBI_WINDOW = 12
RECONNECT_INITIAL_SECONDS = 2
RECONNECT_MAX_SECONDS = 30


class LivePredictionService:
    """Thread-safe live order-book service with reconnect and source selection."""

    def __init__(self, url=DEFAULT_URL):
        self.predictor = Rolling_prediction()
        self.lock = threading.RLock()
        self.lifecycle_lock = threading.Lock()
        self.websocket_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.websocket = None
        self.url = self._validate_url(url)
        self.obi_values = deque(maxlen=OBI_WINDOW)
        self.prediction_history = deque(maxlen=5)
        self.prediction = None
        self.probabilities = None
        self.error = None
        self.connection_status = "Not started"
        self.messages_received = 0
        self.valid_book_messages = 0
        self.last_message_at = None
        self.last_close_code = None
        self.reconnect_attempt = 0

    @staticmethod
    def _validate_url(url):
        url = str(url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme != "wss" or not parsed.netloc:
            raise ValueError("WebSocket URL must be a valid wss:// URL.")
        return url

    @staticmethod
    def _get_book(data):
        """Return complete bids/asks snapshot arrays from supported payloads."""
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # Binance partial-depth snapshots and compatible exchange payloads.
        bids = data.get("bids") if isinstance(data, dict) else None
        asks = data.get("asks") if isinstance(data, dict) else None

        if not bids or not asks:
            raise ValueError(
                "Unsupported message format: expected a complete order-book "
                "snapshot containing non-empty 'bids' and 'asks' arrays."
            )

        parsed_bids = []
        parsed_asks = []
        for side, levels, output in (
            ("bids", bids, parsed_bids),
            ("asks", asks, parsed_asks),
        ):
            for level in levels:
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    raise ValueError(f"Invalid {side} level; expected [price, quantity].")
                price, quantity = float(level[0]), float(level[1])
                if price <= 0 or quantity < 0:
                    raise ValueError(f"Invalid {side} price or quantity.")
                output.append((price, quantity))

        if not parsed_bids or not parsed_asks:
            raise ValueError("Order-book snapshot has no usable bid or ask levels.")
        return parsed_bids, parsed_asks

    def start(self, *, reset=False, url=None):
        """Start feed; optionally change source and/or reset prediction history."""
        with self.lifecycle_lock:
            if self.thread is not None and self.thread.is_alive():
                return

            if url is not None:
                self.url = self._validate_url(url)

            if reset:
                with self.lock:
                    self.predictor.window.clear()
                    self.obi_values.clear()
                    self.prediction_history.clear()
                    self.prediction = None
                    self.probabilities = None
                    self.error = None
                    self.messages_received = 0
                    self.valid_book_messages = 0
                    self.last_message_at = None
                    self.last_close_code = None

            self.stop_event.clear()
            with self.lock:
                self.connection_status = "Connecting"
                self.reconnect_attempt = 0
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _on_message(self, ws, message):
        if self.stop_event.is_set():
            return

        with self.lock:
            self.messages_received += 1
            self.last_message_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        try:
            data = json.loads(message)
            bids, asks = self._get_book(data)

            bid_quantity = sum(quantity for _, quantity in bids) / len(bids)
            ask_quantity = sum(quantity for _, quantity in asks) / len(asks)
            total_quantity = bid_quantity + ask_quantity
            obi = (
                (bid_quantity - ask_quantity) / total_quantity
                if total_quantity else 0.0
            )

            with self.lock:
                if self.stop_event.is_set():
                    return

                self.valid_book_messages += 1
                result = self.predictor.predict(obi)
                if self.stop_event.is_set():
                    return

                self.obi_values.append(float(obi))
                self.error = None
                if result is not None:
                    prediction, probabilities = result
                    self.prediction = prediction[-1]
                    self.probabilities = [float(p) for p in probabilities[-1]]
                    self.prediction_history.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        ),
                        "prediction": self.prediction,
                        "probabilities": list(self.probabilities),
                    })
        except Exception as exc:
            with self.lock:
                self.error = f"Message processing error: {exc}"

    def _on_error(self, ws, error):
        with self.lock:
            self.connection_status = "Connection error"
            self.error = f"WebSocket error: {error!r}"

    def _on_close(self, ws, close_status_code, close_msg):
        with self.lock:
            self.last_close_code = close_status_code
            if not self.stop_event.is_set():
                self.connection_status = "Disconnected"
                self.error = (
                    f"WebSocket closed (code={close_status_code}, "
                    f"reason={close_msg!r}). Reconnecting..."
                )

    def _on_open(self, ws):
        if self.stop_event.is_set():
            ws.close()
            return
        with self.lock:
            self.connection_status = "Connected"
            self.error = None
            self.reconnect_attempt = 0

    def _run(self):
        delay = RECONNECT_INITIAL_SECONDS
        while not self.stop_event.is_set():
            with self.lock:
                self.connection_status = "Connecting"

            current_websocket = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            with self.websocket_lock:
                self.websocket = current_websocket

            try:
                # Heartbeats help detect dead connections and keep intermediaries alive.
                current_websocket.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                    http_no_proxy=["stream.binance.com"],
                )
            except Exception as exc:
                with self.lock:
                    self.connection_status = "Connection error"
                    self.error = f"WebSocket run_forever exception: {exc!r}"
            finally:
                with self.websocket_lock:
                    if self.websocket is current_websocket:
                        self.websocket = None

            if self.stop_event.is_set():
                break

            with self.lock:
                self.reconnect_attempt += 1
                self.connection_status = "Reconnecting"
                if self.error is None:
                    self.error = "Connection ended without a close reason; retrying."

            self.stop_event.wait(delay)
            delay = min(delay * 2, RECONNECT_MAX_SECONDS)

        with self.lock:
            if self.stop_event.is_set():
                self.connection_status = "Stopped"

    def snapshot(self):
        with self.lock:
            return {
                "running": (
                    self.thread is not None
                    and self.thread.is_alive()
                    and not self.stop_event.is_set()
                ),
                "url": self.url,
                "prediction": self.prediction,
                "probabilities": list(self.probabilities)
                if self.probabilities is not None else None,
                "obi_values": list(self.obi_values),
                "prediction_history": [
                    {**record, "probabilities": list(record["probabilities"])}
                    for record in self.prediction_history
                ],
                "error": self.error,
                "connection_status": self.connection_status,
                "messages_received": self.messages_received,
                "valid_book_messages": self.valid_book_messages,
                "last_message_at": self.last_message_at,
                "last_close_code": self.last_close_code,
                "obi_count": len(self.obi_values),
                "reconnect_attempt": self.reconnect_attempt,
            }

    def stop(self):
        with self.lifecycle_lock:
            self.stop_event.set()
            with self.lock:
                self.connection_status = "Stopped"
            with self.websocket_lock:
                current_websocket = self.websocket
                self.websocket = None
            thread = self.thread

            if current_websocket is not None:
                try:
                    current_websocket.close()
                except Exception as exc:
                    with self.lock:
                        self.error = f"Error closing WebSocket: {exc!r}"

            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=5)

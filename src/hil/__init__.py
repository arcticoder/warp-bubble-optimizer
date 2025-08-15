from __future__ import annotations

class HILInterface:
    """Minimal HIL I/O contract for loopback smoke tests."""
    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def write(self, channel: str, value: float):
        self.inputs[channel] = float(value)

    def read(self, channel: str) -> float:
        return float(self.outputs.get(channel, 0.0))

    def loopback(self):
        # Mirror inputs to outputs for smoke tests
        self.outputs.update(self.inputs)

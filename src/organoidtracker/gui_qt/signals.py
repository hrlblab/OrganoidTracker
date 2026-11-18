"""Deliver process stop requests through Qt's ordinary cooperative close path."""

import signal

from PySide6.QtCore import QTimer


class StopBridge:
    def __init__(self, window):
        self.window = window
        self.requested = False
        self.previous = {}
        self.timer = QTimer(window)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.deliver)

    def request(self, signum=None, frame=None):
        self.requested = True

    def deliver(self):
        if self.requested:
            self.requested = False
            self.window.close()

    def install(self):
        for number in (signal.SIGTERM, signal.SIGINT):
            self.previous[number] = signal.signal(number, self.request)
        self.timer.start()

    def restore(self):
        self.timer.stop()
        for number, handler in self.previous.items():
            signal.signal(number, handler)

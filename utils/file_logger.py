# utils/file_logger.py

import os
import sys
import time

class FileLogger:
    def __init__(self, log_dir="logs", filename=None, stdout=True):
        os.makedirs(log_dir, exist_ok=True)

        if filename is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"log_{timestamp}.txt"

        self.log_path = os.path.join(log_dir, filename)
        self.stdout = stdout

        self.file = open(self.log_path, 'w')
        self._write(f"[FileLogger] Logging to {self.log_path}")

    def _write(self, message):
        self.file.write(message + "\n")
        self.file.flush()
        if self.stdout:
            print(message)

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"[{timestamp}] {message}")

    def close(self):
        self._write("[FileLogger] Closing logger.")
        self.file.close()

    def __del__(self):
        if not self.file.closed:
            self.close()

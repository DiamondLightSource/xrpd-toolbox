import os
from datetime import datetime
from pathlib import Path


class AnalysisLogger:
    def __init__(self, log_filepath: str | Path, logging=False):
        self.log_filepath = log_filepath
        self.logging = logging

        if not os.path.exists(self.log_filepath):
            os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        elif os.path.exists(self.log_filepath) and (
            os.path.getsize(self.log_filepath) > 1e7
        ):
            os.remove(self.log_filepath)
            with open(self.log_filepath, "a+") as f:
                f.write("Log File for I11 Data Reduction\n")

        with open(self.log_filepath, "a+") as f:
            f.write("================================\n")
            f.write(f"Datetime: {datetime.now()}\n")
            f.write("================================\n")

    def log(self, *args, print_to_console=True):
        if print_to_console:
            print(*args)

        if self.logging:
            with open(self.log_filepath, "a") as f:
                [f.write(str(m)) for m in args]
                f.write("\n")

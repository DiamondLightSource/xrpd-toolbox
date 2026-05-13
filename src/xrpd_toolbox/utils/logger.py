from datetime import datetime
from pathlib import Path


class AnalysisLogger:
    def __init__(
        self, log_filepath: str | Path, log_to_file: bool = False, beamline: str = "i11"
    ):
        self.log_filepath = Path(log_filepath)
        self.log_to_file = log_to_file

        if not self.log_filepath.exists():
            self.log_filepath.parent.mkdir(parents=True, exist_ok=True)
        elif self.log_filepath.exists() and (self.log_filepath.stat().st_size > 1e7):
            self.log_filepath.unlink()
            with open(self.log_filepath, "a+") as f:
                f.write(f"Log File for {beamline} Data Reduction\n")

        with open(self.log_filepath, "a+") as f:
            f.write("================================\n")
            f.write(f"Datetime: {datetime.now()}\n")
            f.write("================================\n")

    def log(self, *args, print_to_console=True):
        if print_to_console:
            print(*args)

        if self.log_to_file:
            with open(self.log_filepath, "a") as f:
                [f.write(str(m)) for m in args]
                f.write("\n")

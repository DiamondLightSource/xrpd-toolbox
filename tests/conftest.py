import os

# Run Qt without a display (GUI tests, and on CI) - must be set before
# pytest-qt makes the QApplication
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

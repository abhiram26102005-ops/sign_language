import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main():
    try:
        from src.real_time_landmark_detection import main as predict
    except ImportError as error:
        raise SystemExit(
            f"Missing dependency: {error}. Install requirements with "
            "python -m pip install -r requirements.txt."
        ) from error
    predict()


if __name__ == "__main__":
    main()
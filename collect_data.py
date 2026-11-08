import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main():
    try:
        from src.collect_landmark_data import main as collect
    except ImportError as error:
        raise SystemExit(
            f"Missing dependency: {error}. Install requirements with "
            "python -m pip install -r requirements.txt."
        ) from error
    collect()


if __name__ == "__main__":
    main()
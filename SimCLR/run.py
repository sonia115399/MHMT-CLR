import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mhmt.pretrain import pretrain_main

if __name__ == "__main__":
    pretrain_main("clr")

from pathlib import Path


RAW_DIR = Path("data/raw/physionet.org/chb02")

files = sorted(RAW_DIR.glob("*.edf"))

print("EDF recordings:", len(files))

for path in files:
    print(path.name)

assert len(files) == 36

print("\nAll CHB02 recordings found.")

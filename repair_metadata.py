import json
from pathlib import Path

root = Path(r"D:\LungCancerAI\data\processed")

broken = []

print("=" * 70)
print("Scanning processed dataset...")
print("=" * 70)

for patient in sorted(root.iterdir()):

    if not patient.is_dir():
        continue

    meta = patient / "metadata.json"

    if not meta.exists():
        print(f"[MISSING] {patient.name}")
        broken.append(patient.name)
        continue

    try:
        with open(meta, "r") as f:
            json.load(f)

    except Exception:
        print(f"[CORRUPTED] {patient.name}")
        broken.append(patient.name)

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print("Broken Patients :", len(broken))

if len(broken):

    print()

    for p in broken:
        print(p)

else:

    print("Everything looks good.")
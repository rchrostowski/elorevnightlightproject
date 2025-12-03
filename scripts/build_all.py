# scripts/build_all.py

import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure root directory is on PYTHONPATH so "src" imports work
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("Using project root:", ROOT)

from src.build_panel import build_panel_firms_with_brightness

# ------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------
print("🌙 Merging HQ mapping, night-lights, and returns...")
df = build_panel_firms_with_brightness()

print("\nPreview of merged firm × county × lights × returns panel:")
print(df.head())

print("\n✅ All data built successfully.")

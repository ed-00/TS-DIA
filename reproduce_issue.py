from lhotse import CutSet
import sys

try:
    path = "cache/aishell4/test/cuts_windowed.jsonl.gz"
    print(f"Loading {path}")
    cuts = CutSet.from_file(path)
    print("Success")
    for cut in cuts:
        pass
    print("Iterated successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

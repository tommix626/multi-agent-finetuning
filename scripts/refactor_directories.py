#!/usr/bin/env python3
import os
import sys

def rename_subdirectories(parent_dir):
    # list only immediate subdirectories
    entries = [e for e in os.scandir(parent_dir) if e.is_dir()]
    # sort by name so your “cluster#0, cluster#1…” order is preserved
    entries.sort(key=lambda e: e.name)

    for i, entry in enumerate(entries):
        new_name = f"adapter_for_expert_{i}"
        new_path = os.path.join(parent_dir, new_name)
        print(f"Renaming {entry.name!r} → {new_name!r}")
        os.rename(entry.path, new_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <parent_directory>")
        sys.exit(1)
    parent = sys.argv[1]
    if not os.path.isdir(parent):
        print(f"Error: {parent!r} is not a directory.")
        sys.exit(1)
    rename_subdirectories(parent)

"""
clear_outputs.py
----------------
Clears all cell outputs from every Jupyter notebook (.ipynb)
in the current directory and all subdirectories.

Usage:
    python clear_outputs.py
"""

import json
import os
import glob

def clear_notebook_outputs(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            if cell.get('outputs') or cell.get('execution_count') is not None:
                cell['outputs'] = []
                cell['execution_count'] = None
                changed = True

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✔ Cleared: {filepath}")
    else:
        print(f"— Skipped (already clean): {filepath}")

if __name__ == '__main__':
    notebooks = glob.glob('./**/*.ipynb', recursive=True)

    if not notebooks:
        print("No .ipynb files found in the current directory or subdirectories.")
    else:
        print(f"Found {len(notebooks)} notebook(s)...\n")
        for nb_path in notebooks:
            clear_notebook_outputs(nb_path)
        print("\nDone.")

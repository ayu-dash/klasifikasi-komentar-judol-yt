import os
import shutil

# Config
DIRS = {
    'datasets': ['.csv'],
    'notebooks': ['.ipynb'],
    'src': ['.py', '.json'] # config.json typically goes with code or separate config folder
}

# Specific mappings to handle ambiguity (e.g. dont move venv)
EXCLUDE_DIRS = ['venv', 'not use', '__pycache__', '.git', '.ipynb_checkpoints', '.mypy_cache']

def reorganize():
    # Create dirs
    for d in DIRS:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created {d}/")

    # Move files
    for filename in os.listdir('.'):
        if os.path.isdir(filename):
            continue
            
        _, ext = os.path.splitext(filename)
        
        # Determine target
        target_dir = None
        for d, exts in DIRS.items():
            if ext in exts:
                target_dir = d
                break
        
        if target_dir:
            # Move
            shutil.move(filename, os.path.join(target_dir, filename))
            print(f"Moved {filename} -> {target_dir}/")

    print("\nFiles moved. Now updating script paths...")
    
    # Update master_labeling.py paths
    master_path = 'src/master_labeling.py'
    if os.path.exists(master_path):
        with open(master_path, 'r') as f:
            content = f.read()
        
        # Update Input/Output paths to point to ../datasets/
        new_content = content.replace(
            "INPUT_FILE = 'comments_from_scraping_new.csv'",
            "INPUT_FILE = '../datasets/comments_from_scraping_new.csv'"
        ).replace(
            "OUTPUT_FILE = 'comments_labeled_final.csv'",
            "OUTPUT_FILE = '../datasets/comments_labeled_final.csv'"
        )
        
        # Update auto_labeling import if needed (it is in same dir, so import works)
        # But if we run from root, we execute `python src/master_labeling.py`
        # which means current WD is root. file path stays relative to WD or script?
        # Usually checking `os.getcwd`. 
        # Best practice: Use absolute path relative to script location.
        
        path_fix = """
import os
import sys
# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
"""
        if "import pandas as pd" in new_content:
             new_content = new_content.replace("import pandas as pd", path_fix + "\nimport pandas as pd")

        with open(master_path, 'w') as f:
            f.write(new_content)
        print("Updated master_labeling.py paths.")

    # Update auto_labeling.py default path
    auto_path = 'src/auto_labeling.py'
    if os.path.exists(auto_path):
        with open(auto_path, 'r') as f:
            content = f.read()
        
        new_content = content.replace(
            "pd.read_csv('comments_from_scraping_new.csv')",
            "pd.read_csv('../datasets/comments_from_scraping_new.csv')"
        )
        with open(auto_path, 'w') as f:
            f.write(new_content)
        print("Updated auto_labeling.py paths.")

if __name__ == "__main__":
    reorganize()

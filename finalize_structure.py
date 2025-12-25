import os
import shutil

def finalize():
    # Define structure
    utils_dir = 'src/utils'
    archive_dir = 'src/archive'
    
    for d in [utils_dir, archive_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created {d}/")
            
    # Init for packages
    with open(os.path.join(utils_dir, '__init__.py'), 'w') as f:
        pass

    # Move Modules
    shutil.move('src/auto_labeling.py', os.path.join(utils_dir, 'auto_labeling.py'))
    if os.path.exists('src/config.json'):
         shutil.move('src/config.json', os.path.join(utils_dir, 'config.json'))
    print("Moved modules to utils/")

    # Move Archive
    archive_files = ['tes.py', 'featuring.py', 'labeling.py', 'reorganize.py']
    for f in archive_files:
        path = os.path.join('src', f)
        if os.path.exists(path):
            shutil.move(path, os.path.join(archive_dir, f))
    print("Moved unused files to archive/")

    # Update master_labeling.py import
    master_path = 'src/master_labeling.py'
    with open(master_path, 'r') as f:
        content = f.read()
    
    # It currently does: from auto_labeling import classify
    # Change to: from utils.auto_labeling import classify
    if "from auto_labeling import classify" in content:
        new_content = content.replace("from auto_labeling import classify", "from utils.auto_labeling import classify")
        with open(master_path, 'w') as f:
            f.write(new_content)
        print("Updated master_labeling.py imports.")

if __name__ == "__main__":
    finalize()

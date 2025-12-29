import json
import os
import re

NOTEBOOK_PATH = 'notebooks/notebook uas.ipynb'
LABELING_SCRIPT_PATH = 'src/labeling.py'

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def create_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source_lines]
    }

def create_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in source_lines]
    }

def extract_sections(script_content):
    sections = {}
    
    # Imports
    imports_match = re.search(r'^(import.*?)(?=\n# ===)', script_content, re.DOTALL | re.MULTILINE)
    if imports_match:
        sections['imports'] = imports_match.group(1).strip().split('\n')
    
    # Common lookahead for next section or end of file
    # We look for the specific header pattern of the next section
    # The header pattern is: # ===...\n# TITLE
    
    # Config
    config_match = re.search(r'(# ===+\n# CONFIGURATION.*?)(?=\n# ===+\n# TEXT NORMALIZER)', script_content, re.DOTALL)
    if config_match:
        content = config_match.group(1).strip().split('\n')
        # Fix paths for notebook
        new_content = []
        for line in content:
            if 'BASE_DIR =' in line:
                new_content.append("    # BASE_DIR setup for notebook")
                new_content.append("    BASE_DIR = '..'")
            elif 'DATASET_DIR =' in line:
                new_content.append("    DATASET_DIR = os.path.join(BASE_DIR, 'datasets')")
            else:
                new_content.append(line)
        sections['config'] = new_content

    # Text Normalizer
    normalizer_match = re.search(r'(# ===+\n# TEXT NORMALIZER.*?)(?=\n# ===+\n# PATTERN MATCHER)', script_content, re.DOTALL)
    if normalizer_match:
        sections['normalizer'] = normalizer_match.group(1).strip().split('\n')

    # Pattern Matcher
    matcher_match = re.search(r'(# ===+\n# PATTERN MATCHER.*?)(?=\n# ===+\n# JUDOL CLASSIFIER)', script_content, re.DOTALL)
    if matcher_match:
        sections['matcher'] = matcher_match.group(1).strip().split('\n')

    # Judol Classifier
    classifier_match = re.search(r'(# ===+\n# JUDOL CLASSIFIER.*?)(?=\n# ===+\n# LABELING PIPELINE)', script_content, re.DOTALL)
    if classifier_match:
        sections['classifier'] = classifier_match.group(1).strip().split('\n')

    # Labeling Pipeline
    # Capture until end of file or main block if exists
    pipeline_match = re.search(r'(# ===+\n# LABELING PIPELINE.*)', script_content, re.DOTALL)
    if pipeline_match:
        pipeline_content = pipeline_match.group(1).strip().split('\n')
        # Remove any  if __name__ == "__main__": blocks
        clean_pipeline = []
        for line in pipeline_content:
            if 'if __name__ ==' in line:
                break
            clean_pipeline.append(line)
        sections['pipeline'] = clean_pipeline
        
    return sections

def main():
    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print(f"Reading {LABELING_SCRIPT_PATH}...")
    script_content = read_file(LABELING_SCRIPT_PATH)
    
    sections = extract_sections(script_content)
    
    new_cells = []
    
    # 1. Header
    new_cells.append(create_markdown_cell([
        "## 1.1 Labeling Pipeline Classes",
        "The following classes contain the logic for normalizing text, matching patterns, and classifying comments.",
        "They are integrated directly from `src/labeling.py` for portability."
    ]))
    
    # 2. Imports & Config
    if 'config' in sections:
        cell_content = ["import os", "import re", "import unicodedata", "import pandas as pd", "import numpy as np", "from tqdm import tqdm"] + sections['config']
        code_cell = create_code_cell(cell_content)
        code_cell['metadata']['tags'] = ['labeling_config'] # Tag for future ref
        new_cells.append(code_cell)
    
    # 3. Classes
    if 'normalizer' in sections:
        new_cells.append(create_code_cell(sections['normalizer']))
    if 'matcher' in sections:
        new_cells.append(create_code_cell(sections['matcher']))
    if 'classifier' in sections:
        new_cells.append(create_code_cell(sections['classifier']))
    if 'pipeline' in sections:
        new_cells.append(create_code_cell(sections['pipeline']))
        
    # 4. Execution Block
    execution_code = [
        "# ==========================================",
        "# EXECUTION: Run Labeling Pipeline",
        "# ==========================================",
        "# This block checks if the labeled file exists. If not, it runs the pipeline.",
        "",
        "output_file = Config.DEFAULT_OUTPUT_FILE",
        "input_file = Config.DEFAULT_INPUT_FILE",
        "",
        "if os.path.exists(output_file):",
        "    print(f\"Labeled file found at: {output_file}\")",
        "    print(\"Skipping labeling process to save time. To re-run, delete the output file or set force_run=True.\")",
        "    final_df = pd.read_csv(output_file)",
        "else:",
        "    print(\"Labeled file NOT found. Starting labeling pipeline...\")",
        "    # Instantiate classes",
        "    normalizer = TextNormalizer()",
        "    matcher = PatternMatcher(normalizer)",
        "    classifier = JudolClassifier(normalizer, matcher)",
        "    pipeline = LabelingPipeline(classifier)",
        "    ",
        "    # Run Pipeline",
        "    try:",
        "        # 1. Load",
        "        df = pipeline.load_data(input_file)",
        "        ",
        "        # 2. Initial Label",
        "        df = pipeline.apply_initial_labels(df)",
        "        ",
        "        # 3. Heuristic Cleaning",
        "        df, mask_expert = pipeline.apply_heuristic_cleaning(df)",
        "        ",
        "        # 4. Train Model",
        "        model = pipeline.train_model(df)",
        "        ",
        "        # 5. Apply Final Labels",
        "        final_df = pipeline.apply_final_labels(df, model, mask_expert)",
        "        ",
        "        # 6. Save",
        "        # pipeline.save_results(final_df, output_file) # Method missing in partial view? doing manual save",
        "        print(f\"Saving results to {output_file}...\")",
        "        final_df.to_csv(output_file, index=False)",
        "        print(\"Done!\")",
        "        ",
        "    except Exception as e:",
        "        print(f\"An error occurred during labeling: {e}\")",
        "        # Fallback if processing fails, ensuring notebook continues if possible",
        "        final_df = pd.DataFrame()"
    ]
    new_cells.append(create_code_cell(execution_code))
    
    # CLEANUP: Remove previously inserted cells to prevent duplicates
    # We identify them by unique strings in their source
    markers = [
        "## 1.1 Labeling Pipeline Classes",
        "# TEXT NORMALIZER (SRP: Text Processing)",
        "# PATTERN MATCHER (SRP: Pattern Detection",
        "# JUDOL CLASSIFIER (SRP: Classification Logic)",
        "# LABELING PIPELINE (SRP: ML Pipeline",
        "# EXECUTION: Run Labeling Pipeline",
        "class Config:\n    \"\"\"Centralized configuration constants.\"\"\""
    ]
    
    cleaned_cells = []
    removed_count = 0
    for cell in notebook['cells']:
        source_str = "".join(cell.get('source', []))
        is_duplicate = False
        for marker in markers:
            if marker in source_str:
                is_duplicate = True
                break
        if not is_duplicate:
            cleaned_cells.append(cell)
        else:
            removed_count += 1
            
    notebook['cells'] = cleaned_cells
    print(f"Removed {removed_count} existing labeling cells.")

    # Insert after cell index 3 (Imports is index 2 usually)
    insertion_index = 4
    for cell in reversed(new_cells):
        notebook['cells'].insert(insertion_index, cell)
        
    print(f"Inserted {len(new_cells)} cells into notebook.")
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Notebook updated successfully.")

if __name__ == "__main__":
    main()

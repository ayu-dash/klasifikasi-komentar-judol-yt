
import json
import os

NOTEBOOK_PATH = '/home/wtf/Documents/kuliah/kuliah-semester-5/Machine Learning/Tugas Deteksi Komen Judol/notebooks/notebook.ipynb'

def fix_notebook():
    print(f"Reading notebook from {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Note: Cells are in nb['cells'] list.
    # 1. Update Imports (Cell 2) - Index 1 (0-based)
    cell_imports = nb['cells'][1]
    # source is a list of strings in json format usually
    # Join them to make string manipulation easier, then split back if needed or keep as list logic
    
    # Simple helper to handle list of strings for source
    def get_source_str(cell):
        if isinstance(cell['source'], list):
            return "".join(cell['source'])
        return cell['source']

    def set_source_str(cell, new_source):
        # Notebook format usually prefers list of strings with newlines, but string works too usually.
        # Let's split by newline to be nice.
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        # Remove last newline char from last line if it exists to avoid double newline issues? 
        # Actually split keeps newlines if we do it right, but split('\n') removes them.
        # Let's just store as list of lines with \n at end except possibly last one.
        lines = new_source.split('\n')
        cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]

    source_imports = get_source_str(cell_imports)
    if "GlobalAveragePooling1D" not in source_imports:
        print("Adding GlobalAveragePooling1D to imports...")
        new_source_imports = source_imports.replace(
            "from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout, Bidirectional",
            "from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D"
        )
        set_source_str(cell_imports, new_source_imports)

    # 2. Update LabelingPipeline (Cell 7) - Index 6
    cell_pipeline = nb['cells'][6]
    print("Updating LabelingPipeline model architecture...")
    source_pipeline = get_source_str(cell_pipeline)
    
    # Old model definition pattern
    old_model = """        model = Sequential([
            vectorize_layer,
            Embedding(Config.MAX_FEATURES + 1, Config.EMBEDDING_DIM),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])"""

    # New model definition pattern
    new_model = """        model = Sequential([
            vectorize_layer,
            Embedding(Config.MAX_FEATURES + 1, Config.EMBEDDING_DIM),
            GlobalAveragePooling1D(),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])"""
    
    if old_model in source_pipeline:
        new_source_pipeline = source_pipeline.replace(old_model, new_model)
        set_source_str(cell_pipeline, new_source_pipeline)
    else:
        print("WARNING: Old model pattern not found in Cell 7. It might have been updated already or format differs.")
        if "GlobalAveragePooling1D" in source_pipeline:
             print("Model already seems to be updated.")

    # 3. Update Execution Logic (Cell 8) - Index 7
    cell_exec = nb['cells'][7]
    print("Updating execution logic in Cell 8...")
    
    new_exec_source = """# ==========================================
# EXECUTION: Run Labeling Pipeline
# ==========================================
# This block checks if the labeled file exists. If not, it runs the pipeline.

output_file = Config.DEFAULT_OUTPUT_FILE
input_file = Config.DEFAULT_INPUT_FILE

if os.path.exists(output_file):
    print(f"Labeled file found at: {output_file}")
    print("Skipping labeling process to save time. To re-run, delete the output file or set force_run=True.")
    final_df = pd.read_csv(output_file)
else:
    print("Labeled file NOT found. Starting labeling pipeline...")
    # Instantiate classes
    normalizer = TextNormalizer()
    matcher = PatternMatcher(normalizer)
    classifier = JudolClassifier(normalizer, matcher)
    pipeline = LabelingPipeline(classifier)
    
    # Run Pipeline
    try:
        # 1. Load
        df = pipeline.load_data(input_file)
        
        # 2. Initial Label
        df = pipeline.apply_initial_labels(df)
        
        # 3. Heuristic Cleaning
        df, mask_expert = pipeline.apply_heuristic_cleaning(df)
        
        # 4. Train Model
        model = pipeline.train_model(df)
        
        # 5. Apply Final Labels
        final_df = pipeline.apply_final_labels(df, model, mask_expert)
        
        # 6. Save
        pipeline.save_results(final_df, output_file)
        
        # Reload to ensure consistency
        final_df = pd.read_csv(output_file)
        print("Done!")
        
    except Exception as e:
        print(f"An error occurred during labeling: {e}")
        import traceback
        traceback.print_exc()
        # Fallback if processing fails, ensuring notebook continues if possible
        final_df = pd.DataFrame()
"""
    set_source_str(cell_exec, new_exec_source)

    print(f"Saving fixed notebook to {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2) # Keep indentation to minimize diff noise if any
    print("Notebook fix complete.")

if __name__ == "__main__":
    fix_notebook()

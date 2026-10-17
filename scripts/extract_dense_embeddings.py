import subprocess
import torch
import glob
import os
import shutil

def main():
    print("Running pipeline to extract NLI cache...")
    cmd = [
        "python", "run_experiments.py", 
        "--corpus", "data/controlled_N_dense_corpus.jsonl",
        "--output-root", "outputs/temp_dense",
        "--nli-cache-path", "data/controlled_ideo_embeddings_dense.pt"
    ]
    subprocess.run(cmd, check=False)
    
    # Check if the file is generated
    if os.path.exists("data/controlled_ideo_embeddings_dense.pt"):
        print("Successfully generated data/controlled_ideo_embeddings_dense.pt!")
    else:
        print("Failed to generate embeddings. Check the pipeline output.")
        
if __name__ == '__main__':
    main()

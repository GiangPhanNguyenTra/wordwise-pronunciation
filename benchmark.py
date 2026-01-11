import os
import json
import torch
import librosa
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from core import pronunciation_trainer

DATASET_PATH = r"D:\Giang\STUDY AT UIT\HK7\KLTN\models\pronunciation\speechocean762"
SCORES_FILE = os.path.join(DATASET_PATH, "resource", "scores.json")
TEST_SCP_FILE = os.path.join(DATASET_PATH, "test", "wav.scp")
OUTPUT_CSV = "final_thesis_benchmark.csv"
TARGET_SR = 16000

trainer = pronunciation_trainer.getTrainer("en")

def load_test_dataset():
    wav_paths = {}
    if os.path.exists(TEST_SCP_FILE):
        with open(TEST_SCP_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt_id = parts[0]
                    rel_path = parts[1]
                    wav_paths[utt_id] = os.path.join(DATASET_PATH, rel_path)
    
    with open(SCORES_FILE, 'r', encoding='utf-8') as f:
        scores_data = json.load(f)
        
    test_items = []
    for utt_id, path in wav_paths.items():
        if utt_id in scores_data and os.path.exists(path):
            item = scores_data[utt_id]
            test_items.append({
                "id": utt_id,
                "path": path,
                "text": item['text'],
                "human_accuracy": item['accuracy'] * 10.0,
                "human_completeness": item['completeness'] * 10.0,
                "human_fluency": item['fluency'] * 10.0,
                "human_total": item['total'] * 10.0
            })
            
    return test_items

def run_benchmark(num_samples=None):
    dataset = load_test_dataset()
    
    if num_samples is not None and num_samples > 0:
        dataset = dataset[:num_samples]
        print(f"Running benchmark on {len(dataset)} samples...")
    else:
        print(f"Running benchmark on ALL {len(dataset)} samples...")
    
    results = []
    
    for item in tqdm(dataset):
        try:
            audio, _ = librosa.load(item['path'], sr=TARGET_SR)
            tensor = torch.from_numpy(audio).float().unsqueeze(0)
            
            ai_res = trainer.processAudioForGivenText(tensor, item['text'])
            
            results.append({
                "id": item['id'],
                "text": item['text'],
                "human_accuracy": item['human_accuracy'],
                "ai_accuracy": ai_res['pronunciation_accuracy'],
                "human_completeness": item['human_completeness'],
                "ai_completeness": ai_res['completeness_score'],
                "human_fluency": item['human_fluency'],
                "ai_fluency": ai_res['fluency_score'],
                "human_total": item['human_total'],
                "ai_total": ai_res['overall_total']
            })
            
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Benchmark finished. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pronunciation benchmark")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to run (default: all)")
    args = parser.parse_args()
    
    run_benchmark(args.samples)
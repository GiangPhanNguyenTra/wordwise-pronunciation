import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Audio

# Import trực tiếp các module core của bạn
from core import pronunciation_trainer

# ---- CẤU HÌNH ----
OUTPUT_CSV = "benchmark_results_50_samples.csv"
NUM_SAMPLES_TO_TEST = 50  # CHỈ TEST 50 MẪU ĐẦU TIÊN
HUMAN_SCORE_MULTIPLIER = 10.0 # Chuẩn hóa điểm của con người (0-10) về thang 0-100

# ---- KHỞI TẠO CÁC THÀNH PHẦN CẦN THIẾT ----
print("Initializing pronunciation trainer...")
trainer = pronunciation_trainer.getTrainer("en")

# ---- HÀM CHÍNH ĐỂ XỬ LÝ MỘT MẪU DỮ LIỆU ----
def process_sample(sample):
    """
    Xử lý một mẫu dữ liệu từ dataset Hugging Face.
    """
    try:
        # 1. Lấy thông tin cần thiết từ mẫu
        transcript = sample['text']
        # Điểm 'total' là điểm tổng thể của câu do chuyên gia chấm (thang 0-10)
        human_score = sample['total'] 
        audio_array = np.array(sample['audio']['array'], dtype=np.float32)
        
        # Đảm bảo audio có đúng định dạng tensor mà trainer mong đợi
        signal_tensor = torch.from_numpy(audio_array).unsqueeze(0)

        # 2. Lấy điểm của AI
        ai_result = trainer.processAudioForGivenText(signal_tensor, transcript)
        ai_score = ai_result.get('pronunciation_accuracy', 0.0)

        # 3. Chuẩn hóa điểm của con người về thang 0-100 để so sánh
        human_score_normalized = human_score * HUMAN_SCORE_MULTIPLIER

        return {
            "transcript": transcript,
            "human_score": human_score_normalized,
            "ai_score": float(ai_score)
        }
    except Exception as e:
        print(f"Error processing a sample: {e}")
        return None

# ---- CHẠY BENCHMARK ----
def run_benchmark():
    print("Loading SpeechOcean762 dataset from Hugging Face...")
    try:
        # Tải tập test. Lần đầu sẽ mất thời gian tải về máy.
        test_dataset = load_dataset("mispeech/speechocean762", split="test")
        # Chuyển đổi cột audio sang định dạng 16kHz nếu cần
        test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    # Lấy ra 50 mẫu đầu tiên để test
    small_test_set = test_dataset.select(range(NUM_SAMPLES_TO_TEST))
    print(f"Dataset loaded. Running benchmark on {len(small_test_set)} samples...")
    
    results = []
    for sample in tqdm(small_test_set, desc="Benchmarking"):
        result = process_sample(sample)
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nBenchmark for {NUM_SAMPLES_TO_TEST} samples complete. Results saved to {OUTPUT_CSV}")
    else:
        print("No results were generated. Please check for errors.")

if __name__ == "__main__":
    run_benchmark()
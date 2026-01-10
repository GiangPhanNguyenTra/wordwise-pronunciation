import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# File CSV kết quả từ bước benchmark
INPUT_CSV = "benchmark_results_50_samples.csv"

def analyze():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{INPUT_CSV}'.")
        print("Vui lòng chạy file benchmark.py trước để tạo ra file kết quả.")
        return

    # ---- 1. TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ ----
    human_scores = df['human_score']
    ai_scores = df['ai_score']

    # Hệ số tương quan Pearson (Pearson Correlation Coefficient - PCC)
    correlation, _ = pearsonr(human_scores, ai_scores)

    # Sai số tuyệt đối trung bình (Mean Absolute Error - MAE)
    mae = mean_absolute_error(human_scores, ai_scores)

    # Sai số bình phương trung bình gốc (Root Mean Squared Error - RMSE)
    rmse = root_mean_squared_error(human_scores, ai_scores)

    # ---- 2. TẠO BÁO CÁO ----
    print("="*60)
    print("      BÁO CÁO BENCHMARKING MODULE PHÁT ÂM")
    print("="*60)
    print(f"Đã phân tích từ file: {INPUT_CSV}")
    print(f"Tổng số mẫu được đánh giá: {len(df)}")
    print("\n--- CÁC CHỈ SỐ CHẤT LƯỢNG ---\n")
    print(f"Hệ số tương quan Pearson (PCC): {correlation:.4f}")
    print(f"Sai số tuyệt đối trung bình (MAE): {mae:.4f} điểm")
    print(f"Sai số bình phương trung bình gốc (RMSE): {rmse:.4f} điểm")
    print("\n--- DIỄN GIẢI ---\n")
    print("1. Hệ số tương quan Pearson (PCC) đo mức độ đồng thuận giữa AI và con người.")
    print("   - Càng gần +1.0: Rất tốt (AI chấm cao khi người chấm cao, và ngược lại).")
    print("   - Giá trị tham khảo cho các model tốt thường > 0.7")
    print("2. MAE/RMSE đo mức độ sai lệch trung bình về điểm số.")
    print("   - Càng gần 0: Càng tốt (Điểm của AI rất gần với điểm của con người).")
    print("="*60)

    # ---- 3. VẼ BIỂU ĐỒ TÁN XẠ (SCATTER PLOT) ----
    plt.figure(figsize=(10, 8))
    sns.regplot(x='human_score', y='ai_score', data=df, scatter_kws={'alpha':0.6})
    plt.title('So sánh điểm phát âm giữa AI và Con người (50 mẫu)', fontsize=16)
    plt.xlabel('Điểm do Con người chấm (Ground Truth)', fontsize=12)
    plt.ylabel('Điểm do AI chấm (Model Score)', fontsize=12)
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
    # Vẽ đường thẳng y=x để dễ so sánh
    plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Tương quan hoàn hảo (AI = Người)')
    plt.legend()
    
    plot_filename = 'benchmark_scatterplot_50_samples.png'
    plt.savefig(plot_filename)
    print(f"\nBiểu đồ so sánh đã được lưu vào file '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    analyze()
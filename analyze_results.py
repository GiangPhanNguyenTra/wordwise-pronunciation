import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Tắt warning thống kê
warnings.filterwarnings("ignore")

# Cấu hình file
CSV_FILE = "final_thesis_benchmark.csv"
REPORT_IMG = "final_thesis_report.png"

def analyze():
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"[INFO] Đang phân tích file: {CSV_FILE}")
        print(f"[INFO] Tổng số mẫu: {len(df)}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {CSV_FILE}")
        return

    # --- 1. TÍNH TOÁN METRICS ---
    metrics = [
        ("Accuracy", "human_accuracy", "ai_accuracy"),
        ("Completeness", "human_completeness", "ai_completeness"),
        ("Fluency", "human_fluency", "ai_fluency"),
        ("Total Score", "human_total", "ai_total")
    ]

    print("\n" + "="*80)
    print(f"{'METRIC':<15} | {'MAE':<8} | {'RMSE':<8} | {'PCC':<8} | {'Mean Diff':<10}")
    print("-" * 80)

    stats = {}
    for name, h_col, a_col in metrics:
        if h_col not in df.columns or a_col not in df.columns:
            continue
            
        h = df[h_col]
        a = df[a_col]
        mae = mean_absolute_error(h, a)
        rmse = np.sqrt(mean_squared_error(h, a))
        
        # Xử lý PCC
        if np.std(h) == 0 or np.std(a) == 0:
            pcc = 0.0
        else:
            pcc, _ = pearsonr(h, a)
            
        diff = np.mean(a - h)
        stats[name] = pcc
        
        print(f"{name:<15} | {mae:<8.2f} | {rmse:<8.2f} | {pcc:<8.3f} | {diff:<10.2f}")
    print("="*80)

    # --- 2. VẼ BIỂU ĐỒ BÁO CÁO ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Biểu đồ 1: Total Score Correlation
    sns.regplot(x='human_total', y='ai_total', data=df, ax=axes[0, 0], 
                x_jitter=1.5, y_jitter=1.5,
                scatter_kws={'alpha':0.4, 'color': '#3498db', 's': 25}, 
                line_kws={'color':'#e74c3c', 'linewidth': 2})
    axes[0, 0].plot([0, 100], [0, 100], ls="--", c="gray", alpha=0.8, label="Ideal Line")
    axes[0, 0].set_title(f"Total Score Correlation (PCC: {stats.get('Total Score', 0):.2f})", fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel("Human Evaluation")
    axes[0, 0].set_ylabel("AI Prediction")
    axes[0, 0].set_xlim(35, 105); axes[0, 0].set_ylim(35, 105)
    axes[0, 0].legend()

    # Biểu đồ 2: Accuracy Correlation
    sns.regplot(x='human_accuracy', y='ai_accuracy', data=df, ax=axes[0, 1],
                x_jitter=1.5, y_jitter=1.5,
                scatter_kws={'alpha':0.4, 'color': '#2ecc71', 's': 25}, 
                line_kws={'color':'#e74c3c', 'linewidth': 2})
    axes[0, 1].plot([0, 100], [0, 100], ls="--", c="gray", alpha=0.8)
    axes[0, 1].set_title(f"Pronunciation Accuracy (PCC: {stats.get('Accuracy', 0):.2f})", fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel("Human Accuracy")
    axes[0, 1].set_ylabel("AI Accuracy")
    axes[0, 1].set_xlim(35, 105); axes[0, 1].set_ylim(35, 105)

    # Biểu đồ 3: Error Distribution
    error = df['ai_total'] - df['human_total']
    sns.histplot(error, kde=True, ax=axes[1, 0], color='#9b59b6', bins=25, edgecolor='white')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=1.5, label="Zero Error")
    axes[1, 0].set_title(f"Error Distribution (Mean Diff: {np.mean(error):.2f})", fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel("Score Difference (AI - Human)")
    axes[1, 0].legend()

    # Biểu đồ 4: Box Plot Comparison
    if 'human_total' in df.columns and 'ai_total' in df.columns:
        data_melted = df.melt(value_vars=['human_total', 'ai_total'], var_name='Evaluator', value_name='Score')
        data_melted['Evaluator'] = data_melted['Evaluator'].replace({'human_total': 'Human', 'ai_total': 'AI Model'})
        
        sns.boxplot(x='Evaluator', y='Score', data=data_melted, ax=axes[1, 1], 
                    palette="Set3", width=0.5, showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
        axes[1, 1].set_title("Score Range Distribution Comparison", fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(REPORT_IMG, dpi=300)
    print(f"\n[SUCCESS] Biểu đồ báo cáo đã được lưu vào: {REPORT_IMG}")

    # --- 3. TOP OUTLIERS (Để biết còn lỗi gì không) ---
    print("\n--- TOP 5 DEVIATIONS (SAI SỐ LỚN NHẤT) ---")
    df['abs_diff'] = abs(df['ai_total'] - df['human_total'])
    top_diff = df.sort_values('abs_diff', ascending=False).head(5)
    print(top_diff[['id', 'text', 'human_total', 'ai_total', 'abs_diff']].to_string(index=False))

if __name__ == "__main__":
    analyze()
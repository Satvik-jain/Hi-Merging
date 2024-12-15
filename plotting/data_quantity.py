import os
import matplotlib.pyplot as plt

# Data
train_samples = [10, 20, 30, 40, 50]
healthcaremagic_ft = [23.0259, 23.3605, 23.7725, 24.1743, 23.8833]
cmedqa2_ft = [11.6925, 11.8879, 11.7912, 11.7066, 11.8967]
model_soups_hcm = [22.9476, 22.6615, 22.5342, 22.1680, 22.0577]
model_soups_cmedqa2 = [11.6682, 11.5977, 12.1760, 11.7964, 11.9101]
task_arithmetic_hcm = [22.9253, 22.6211, 22.3748, 22.5090, 21.9659]
task_arithmetic_cmedqa2 = [11.6259, 11.6501, 12.1040, 11.8365, 11.9047]
ours_hcm = [23.095, 23.2728, 23.3883, 24.1705, 23.8574]
ours_cmedqa2 = [11.9937, 11.9562, 11.8558, 11.7834, 11.9730]

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Modify the plot function to use legends instead of side labels
def plot_and_save(title, x, y1, y2, label1, label2, filename):

    fig, ax1 = plt.subplots(figsize=(8, 8))

    # 设置X轴刻度文本大小
    plt.xticks(fontsize=32)

    # Plot data for HealthCareMagic
    ax1.plot(x, y1, marker='o', color='tab:blue', label=label1, linewidth=2)
    ax1.set_xlabel('Training Samples (k)', fontsize=32)
    ax1.xaxis.set_label_coords(0.5, -0.1)  # 设置xlabel位置
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=32)
    ax1.set_xticks(x)
    ax1.set_ylim([21, 25])
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot data for cMedQA2 on the second y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, y2, marker='o', color='tab:orange', label=label2, linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:orange', labelsize=32)
    ax2.set_ylim([11, 15])

    # Add legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=26)

    # Save the plot
    plt.savefig(os.path.join('data', filename), bbox_inches='tight')
    plt.close(fig)

# Save plots to 'data' folder
plot_and_save('Performance of Fine-Tuned Models', train_samples, healthcaremagic_ft, cmedqa2_ft,
              'HealthCareMagic', 'cMedQA2', 'Figure 5 (1).pdf')

plot_and_save('Performance of Model Soups', train_samples, model_soups_hcm, model_soups_cmedqa2,
              'HealthCareMagic', 'cMedQA2', 'Figure 5 (2).pdf')

plot_and_save('Performance of Task Arithmetic', train_samples, task_arithmetic_hcm, task_arithmetic_cmedqa2,
              'HealthCareMagic', 'cMedQA2', 'Figure 5 (3).pdf')

plot_and_save('Performance of Our Models', train_samples, ours_hcm, ours_cmedqa2,
              'HealthCareMagic', 'cMedQA2', 'Figure 5 (4).pdf')

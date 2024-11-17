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


# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Modify the plot function to save plots
def plot_and_save(title, x, y1, y2, label1, label2, filename):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # First Y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Training Samples (k)', fontsize=18)
    ax1.set_ylabel(label1, color=color1, fontsize=18)
    ax1.plot(x, y1, marker='o', color=color1, label=label1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax1.set_xticks(x)  # Set x-axis ticks to match data points
    ax1.set_ylim([20, 25])  # Set y-axis range for first axis
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Second Y-axis
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color2 = 'tab:orange'
    ax2.set_ylabel(label2, color=color2, fontsize=18)
    ax2.plot(x, y2, marker='o', color=color2, label=label2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)
    ax2.set_ylim([10, 15])  # Set y-axis range for second axis

    # Title
    # fig.suptitle(title, fontsize=16)

    # Save the plot
    plt.savefig(os.path.join('data', filename), bbox_inches='tight')
    plt.close(fig)

# Save plots to 'data' folder
plot_and_save('Performance of Fine-Tuned Models', train_samples, healthcaremagic_ft, cmedqa2_ft,
              'HealthCareMagic', 'cMedQA2', 'fine_tuned_models.pdf')

plot_and_save('Performance of Model Soups', train_samples, model_soups_hcm, model_soups_cmedqa2,
              'HealthCareMagic', 'cMedQA2', 'model_soups.pdf')

plot_and_save('Performance of Task Arithmetic', train_samples, task_arithmetic_hcm, task_arithmetic_cmedqa2,
              'HealthCareMagic', 'cMedQA2', 'task_arithmetic.pdf')



import numpy as np
import matplotlib.pyplot as plt

# 数据准备
labels = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']
num_vars = len(labels)

# 模型数据
model1 = [90, 85, 80, 75, 70, 50, 50, 50, 50, 50]  # 左边5个表现好
model2 = [50, 50, 50, 50, 50, 70, 75, 80, 85, 90]  # 右边5个表现好
model3 = [85, 80, 85, 80, 85, 85, 80, 85, 80, 85]  # 两边都表现好

# 数据闭合
models = [model1, model2, model3]
for i in range(len(models)):
    models[i] += models[i][:1]  # 闭合曲线

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合角度

# 调整旋转角度，让右边旋转两格（指标从 M9 开始）
angles = np.roll(angles, -2)  # -2 表示顺时针偏移两格
rotated_labels = np.roll(labels, -2)  # 确保标签顺时针对应

# 绘图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制每个模型
colors = ['blue', 'green', 'red']
labels_models = ['Model1', 'Model2', 'Model3']
for model, color, label in zip(models, colors, labels_models):
    ax.fill(angles, model, color=color, alpha=0.25)  # 填充颜色
    ax.plot(angles, model, color=color, linewidth=2, label=label)  # 边界线

# 设置标签和样式
ax.set_xticks(angles[:-1])  # 不包括闭合点的角度
ax.set_xticklabels(rotated_labels, fontsize=10)  # 对应旋转后的标签
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], color='grey', fontsize=8)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))  # 图例

# 美化
ax.spines['polar'].set_visible(False)  # 隐藏边框
ax.grid(color='grey', linestyle='--', linewidth=0.5)
ax.set_title("Performance", fontsize=16)

# 显示
plt.tight_layout()
plt.show()

#!/bin/bash

# ------------------------
# 1) 基础设置
# ------------------------

# 你要处理的yml文件路径
TIES_YML="/data/user/PycharmProjects/mergekit/examples/ties.yml"

# 训练使用的配置文件
LLAMA_YML="/data/user/PycharmProjects/LLaMA-Factory/examples/train_lora/qwen2_lora_predict.yaml"

# 输出目录前缀(循环中不同迭代会自动加后缀)
OUTPUT_DIR_PREFIX="saves/qwen2-7b/lora/predict/en_iteration_"

# MergeKit 和 LLaMA-Factory 的项目根目录
MERGEKIT_DIR="/data/user/PycharmProjects/mergekit"
LLAMAFACTORY_DIR="/data/user/PycharmProjects/LLaMA-Factory"

# 一共有多少个位置要循环(权重数组长度)
NUM_POSITIONS=28

# ------------------------
# 2) 文件备份
# ------------------------
echo "[Info] 备份原始配置..."
cp "$TIES_YML" "${TIES_YML}.orig"
cp "$LLAMA_YML" "${LLAMA_YML}.orig"

# ------------------------
# 3) 找到 en / cmexam 对应的 weight 行号
#    (假设 ties.yml 结构固定, 先定位到模型行,再往下搜 'weight:' 行)
# ------------------------

# 定位 "qwen2_lora_sft/en" 出现的行号
EN_LINE=$(grep -n -m1 'qwen2_lora_sft/en' "$TIES_YML" | cut -d':' -f1)
if [ -z "$EN_LINE" ]; then
  echo "[Error] 未找到 en 模型配置(qwen2_lora_sft/en)"
  exit 1
fi

# 从 EN_LINE 往后搜 'weight:' 的行号
EN_WEIGHT_LINE=$(awk -v mline="$EN_LINE" 'NR > mline && /weight:/ {print NR; exit}' "$TIES_YML")
if [ -z "$EN_WEIGHT_LINE" ]; then
  echo "[Error] 未找到 en 模型的 weight: 行"
  exit 1
fi

# 定位 "qwen2_lora_sft/cmexam" 出现的行号
CMEXAM_LINE=$(grep -n -m1 'qwen2_lora_sft/cmexam' "$TIES_YML" | cut -d':' -f1)
if [ -z "$CMEXAM_LINE" ]; then
  echo "[Error] 未找到 cmexam 模型配置(qwen2_lora_sft/cmexam)"
  exit 1
fi

# 从 CMEXAM_LINE 往后搜 'weight:' 的行号
CMEXAM_WEIGHT_LINE=$(awk -v mline="$CMEXAM_LINE" 'NR > mline && /weight:/ {print NR; exit}' "$TIES_YML")
if [ -z "$CMEXAM_WEIGHT_LINE" ]; then
  echo "[Error] 未找到 cmexam 模型的 weight: 行"
  exit 1
fi

# ------------------------
# 4) 循环修改 ties.yml, 并执行 merge 和训练
# ------------------------
for ((i=1; i<=NUM_POSITIONS; i++)); do
  echo "------------------------------"
  echo "[Info] 开始第 $i 次迭代..."
  echo "------------------------------"

  # 先恢复原始 ties.yml
  cp "${TIES_YML}.orig" "$TIES_YML"

  # ------------------------
  # 4.1) 修改 ties.yml
  # 
  # 第 i 次迭代:
  #  - 把 en 的 weight 数组中第 i 个 "0.0" 替换成 "1.0"
  #  - 把 cmexam 的 weight 数组中第 i 个 "0.0" 替换成 "1.0"
  # 
  # 注意: sed 的 "s/old/new/Nth" 表示替换第 N 次出现的 old
  # ------------------------
  sed -i "${EN_WEIGHT_LINE}s/0.0/0.5/${i}" "$TIES_YML"
  sed -i "${CMEXAM_WEIGHT_LINE}s/0.0/0.5/${i}" "$TIES_YML"

  # ------------------------
  # 4.2) 修改 qwen2_lora_predict.yaml (输出目录)
  # ------------------------
  cp "${LLAMA_YML}.orig" "$LLAMA_YML"
  THIS_OUTPUT_DIR="${OUTPUT_DIR_PREFIX}${i}"
  sed -i "s|^output_dir:.*|output_dir: ${THIS_OUTPUT_DIR}|" "$LLAMA_YML"

  # ------------------------
  # 4.3) 执行 MergeKit
  # ------------------------
  echo "[Info] 执行 MergeKit..."
  cd "$MERGEKIT_DIR" || exit 1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mergekit-yaml \
    ./examples/ties.yml \
    ./output_model/qwen2_lora_sft/merge_en_zh \
    --allow-crimes --cuda

  # ------------------------
  # 4.4) 执行 LLaMA-Factory 训练
  # ------------------------
  echo "[Info] 执行 LLaMA-Factory 训练..."
  cd "$LLAMAFACTORY_DIR" || exit 1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    examples/train_lora/qwen2_lora_predict.yaml

  echo "[Info] 第 $i 次迭代完成。"
done

# ------------------------
# 5) 恢复原始文件
# ------------------------
echo "[Info] 所有迭代完成。恢复原始文件。"
cp "${TIES_YML}.orig" "$TIES_YML"
cp "${LLAMA_YML}.orig" "$LLAMA_YML"

echo "[Done] 脚本执行完毕。"


for density in $(seq 0.1 0.1 1.0); do
  for weight in $(seq 0.1 0.1 1.0); do
    # 切换到 mergekit 目录
    cd /data/user/PycharmProjects/mergekit

    # 修改 density 的值
    sed -i "s/density: .*/density: $density/" ./examples/ties.yml

    # 修改 weight 的值
    sed -i "s/weight: .*/weight: $weight/" ./examples/ties.yml

    # 执行合并命令
    mergekit-yaml ./examples/ties.yml ./output_model/qwen2_lora_sft/merge_en_zh --allow-crimes --cuda

    # 切换到 LLaMA-Factory 目录
    cd /data/user/PycharmProjects/LLaMA-Factory

    # 修改 output_dir 的值
    output_dir="/data/user/PycharmProjects/LLaMA-Factory/saves/qwen2-7b/lora/predict/merge-p-en-${density}-${weight}"
    sed -i "s|output_dir:.*|output_dir: $output_dir|" ./examples/train_lora/qwen2_lora_predict.yaml

    # 执行训练命令
    llamafactory-cli train ./examples/train_lora/qwen2_lora_predict.yaml
  done
done

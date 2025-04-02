for density in $(seq 0.1 0.1 1.0); do
  for weight in $(seq 0.1 0.1 1.0); do
    # Change directory to mergekit
    cd /data/user/PycharmProjects/mergekit

    # Update the value of 'density'
    sed -i "s/density: .*/density: $density/" ./examples/ties.yml

    # Update the value of 'weight'
    sed -i "s/weight: .*/weight: $weight/" ./examples/ties.yml

    # Execute merge command
    mergekit-yaml ./examples/ties.yml ./output_model/qwen2_lora_sft/merge_en_zh --allow-crimes --cuda

    # Change directory to LLaMA-Factory
    cd /data/user/PycharmProjects/LLaMA-Factory

    # Update the 'output_dir' value
    output_dir="/data/user/PycharmProjects/LLaMA-Factory/saves/qwen2-7b/lora/predict/merge-p-en-${density}-${weight}"
    sed -i "s|output_dir:.*|output_dir: $output_dir|" ./examples/train_lora/qwen2_lora_predict.yaml

    # Run the training command
    llamafactory-cli train ./examples/train_lora/qwen2_lora_predict.yaml
  done
done

from awq import AwqConfig

# 양자화 설정 변환
quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size=quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"].lower()
).to_dict()

# 설정 저장
model.model.config.quantization_config = quantization_config

# 모델 저장
model.save_pretrained("quantized_model_output_path")[8]

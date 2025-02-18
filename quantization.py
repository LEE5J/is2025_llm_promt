import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "Nexusflow/Athene-V2-Agent"
quant_path = "lee5j/Athene-V2-Agent-gptqmodel-4bit"
calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=10)

model.save(quant_path)

# test post-quant inferencepip install --upgrade wheel setuptools pip

result = model.generate("Uncovering deep insights begins with")[0]
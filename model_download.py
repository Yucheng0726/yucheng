from transformers import AutoModel

# 加载模型
model = AutoModel.from_pretrained("Hhhhhao97/ldm_imagenet_random_noise_2.5")

# 保存到指定路径
model.save_pretrained("runwayml/pretrainmodel")

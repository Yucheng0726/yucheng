from diffusers import StableDiffusionControlNetPipeline

# 加载预训练模型
model_id = "Hhhhhao97/ldm_imagenet_random_noise_2.5"
pipeline = StableDiffusionControlNetPipeline.from_pretrained(model_id)

# 保存模型到指定路径
save_path = "runwayml/pretrainmodel"
pipeline.save_pretrained(save_path)

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UniDiffuserModel, UniDiffuserTextDecoder
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from pipelines.pipeline_jodiffusion import JoDiffusionPipeline
from pipelines.modeling_uvit import JoDiffusionModel

image_vae = AutoencoderKL.from_pretrained("thu-ml/unidiffuser-v1", subfolder="vae")
label_vae = AutoencoderKL.from_pretrained("thu-ml/unidiffuser-v1", subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained("thu-ml/unidiffuser-v1", subfolder="text_encoder")
clip_tokenizer = CLIPTokenizer.from_pretrained("thu-ml/unidiffuser-v1", subfolder="clip_tokenizer")
image_encoder = CLIPVisionModelWithProjection.from_pretrained("thu-ml/unidiffuser-v1", subfolder="image_encoder")
clip_image_processor = CLIPImageProcessor.from_pretrained("thu-ml/unidiffuser-v1", subfolder="clip_image_processor")
udiff = UniDiffuserModel.from_pretrained("thu-ml/unidiffuser-v1", subfolder="unet")
text_decoder = UniDiffuserTextDecoder.from_pretrained("thu-ml/unidiffuser-v1", subfolder="text_decoder")
unet = JoDiffusionModel()

# use modified unidiffuser model parameters
unet.pre_text = text_decoder.encode_prefix
unet.text_in = udiff.text_in
unet.vae_img_in = udiff.vae_img_in
unet.clip_img_in = udiff.clip_img_in
unet.vae_label_in = udiff.vae_img_in
unet.timestep_img_proj = udiff.timestep_img_proj
unet.timestep_img_embed = udiff.timestep_img_embed
unet.timestep_text_proj = udiff.timestep_text_proj
unet.timestep_text_embed = udiff.timestep_text_embed
# img + text + 77 + 1 + 1024 + 1024 <== img + text + 77 + 1 + 1024
new_pos_embed = torch.nn.Parameter(torch.cat([
    udiff.pos_embed, udiff.pos_embed[:, -1024:]
], dim=1))
unet.pos_embed = new_pos_embed
unet.data_type_pos_embed_token = udiff.data_type_pos_embed_token
unet.data_type_token_embedding = udiff.data_type_token_embedding
unet.transformer = udiff.transformer
unet.text_out = udiff.text_out
unet.post_text = text_decoder.decode_prefix
unet.vae_img_out = udiff.vae_img_out
unet.clip_img_out = udiff.clip_img_out
unet.label_out = udiff.vae_img_out

scheduler = DPMSolverMultistepScheduler.from_pretrained("thu-ml/unidiffuser-v1", subfolder="scheduler")
pipe = JoDiffusionPipeline(
    image_vae=image_vae,
    text_encoder=text_encoder,
    clip_tokenizer=clip_tokenizer,
    image_encoder=image_encoder,
    clip_image_processor=clip_image_processor,
    label_vae=label_vae,
    unet=unet,
    scheduler=scheduler,
)
pipe = pipe.to('cuda', dtype=torch.float32)

# Unconditional image and text generation. The generation task is automatically inferred.
sample = pipe(prompt="cat", num_inference_steps=20, guidance_scale=8.0, num_images_per_prompt=1)
image = sample.images[0]
label = sample.labels[0]
image.save("jodiffusion_joint_sample_image.png")
label.save("jodiffusion_joint_sample_label.png")

pipe.save_pretrained("inference/saved_pipeline/jodiffusion", safe_serialization=False)

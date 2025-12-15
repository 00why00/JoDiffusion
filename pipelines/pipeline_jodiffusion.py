import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from PIL import Image
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from utils.utils import encode_seg

from .modeling_uvit import JoDiffusionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# New BaseOutput child class for joint image-label output
@dataclass
class ImageLabelPipelineOutput(BaseOutput):
    """
    Output class for joint image-text pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        labels (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL labels of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Optional[Union[List[Image.Image], np.ndarray]]
    labels: Optional[Union[List[Image.Image], np.ndarray]]


class JoDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for a bimodal image-label model which supports unconditional joint image-label generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        image_vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations. This
            is part of the UniDiffuser image representation along with the CLIP vision encoding.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        clip_tokenizer ([`CLIPTokenizer`]):
             A [`~transformers.CLIPTokenizer`] to tokenize the prompt before encoding it with `text_encoder`.
        image_encoder ([`CLIPVisionModel`]):
            A [`~transformers.CLIPVisionModel`] to encode images as part of its image representation along with the VAE
            latent representation.
        clip_image_processor ([`CLIPImageProcessor`]):
            [`~transformers.CLIPImageProcessor`] to preprocess an image before CLIP encoding it with `image_encoder`.
        label_vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode labels as part of its label representation
            along with the CLIP latent representation.
        unet ([`JoDiffusionModel`]):
            A [U-ViT](https://github.com/baofff/U-ViT) model with UNNet-style skip connections between transformer
            layers to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image and/or text latents. The
            original UniDiffuser paper uses the [`DPMSolverMultistepScheduler`] scheduler.
    """
    model_cpu_offload_seq = "image_encoder->unet->image_vae->label_vae"

    def __init__(
        self,
        image_vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        clip_tokenizer: CLIPTokenizer,
        image_encoder: CLIPVisionModelWithProjection,
        clip_image_processor: CLIPImageProcessor,
        label_vae: AutoencoderKL,
        unet: JoDiffusionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(
            image_vae=image_vae,
            text_encoder=text_encoder,
            clip_tokenizer=clip_tokenizer,
            image_encoder=image_encoder,
            clip_image_processor=clip_image_processor,
            label_vae=label_vae,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.image_vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.num_channels_latents = image_vae.config.latent_channels  # noqa
        self.text_encoder_seq_len = text_encoder.config.max_position_embeddings
        self.text_encoder_hidden_size = text_encoder.config.hidden_size
        self.image_encoder_projection_dim = image_encoder.config.projection_dim
        self.unet_resolution = unet.sample_size

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.image_vae.enable_slicing()
        self.label_vae.enable_slicing()

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.image_vae.disable_slicing()
        self.label_vae.disable_slicing()

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.image_vae.enable_tiling()
        self.label_vae.enable_tiling()

    # Copied from diffusers.pipelines.pipeline_utils.StableDiffusionMixin.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.image_vae.disable_tiling()
        self.label_vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt with self.tokenizer->self.clip_tokenizer and remove lora
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.clip_tokenizer)

            text_inputs = self.clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=self.clip_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.clip_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.clip_tokenizer.batch_decode(
                    untruncated_ids[:, self.clip_tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.clip_tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.clip_tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.clip_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    # Rename prepare_latents -> prepare_image_vae_latents and add num_prompts_per_image argument.
    def prepare_image_vae_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # latents is assumed to have shape (B, C, H, W)
            latents = latents.repeat(batch_size, 1, 1, 1)
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_clip_embeds(
        self, batch_size, clip_img_dim, dtype, device, generator, latents=None
    ):
        # Prepare latents for the CLIP embedded image.
        shape = (batch_size, 1, clip_img_dim)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # latents is assumed to have shape (B, L, D)
            latents = latents.repeat(batch_size, 1, 1)
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_label_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # latents is assumed to have shape (B, C, H, W)
            latents = latents.repeat(batch_size, 1, 1, 1)
            latents = latents.to(device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _split(self, x, height, width):
        r"""
        Splits a flattened embedding x of shape (B, C * H * W + clip_img_dim) into two tensors of shape (B, C, H, W)
        and (B, 1, clip_img_dim)
        """
        batch_size = x.shape[0]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        img_vae_dim = self.num_channels_latents * latent_height * latent_width
        label_dim = self.num_channels_latents * latent_height * latent_width

        img_vae, img_clip, label_clip = x.split([img_vae_dim, self.image_encoder_projection_dim, label_dim], dim=1)

        img_vae = torch.reshape(img_vae, (batch_size, self.num_channels_latents, latent_height, latent_width))
        img_clip = torch.reshape(img_clip, (batch_size, 1, self.image_encoder_projection_dim))
        label = torch.reshape(label_clip, (batch_size, self.num_channels_latents, latent_height, latent_width))
        return img_vae, img_clip, label

    @staticmethod
    def _combine(img_vae, img_clip, label):
        r"""
        Combines a latent iamge img_vae of shape (B, C, H, W) and a CLIP-embedded image img_clip of shape (B, 1,
        clip_img_dim) into a single tensor of shape (B, C * H * W + clip_img_dim).
        """
        img_vae = torch.reshape(img_vae, (img_vae.shape[0], -1))
        img_clip = torch.reshape(img_clip, (img_clip.shape[0], -1))
        label = torch.reshape(label, (label.shape[0], -1))
        return torch.concat([img_vae, img_clip, label], dim=-1)

    def _split_joint(self, x, height, width):
        r"""
        Splits a flattened embedding x of shape (B, C * H * W + clip_img_dim + clip_img_dim] into (img_vae,
        img_clip, label_clip) where img_vae is of shape (B, C, H, W), img_clip is of shape (B, 1, clip_img_dim),
        and label_clip is of shape (B, 1, clip_img_dim).
        """
        batch_size = x.shape[0]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        text_dim = self.text_encoder_seq_len * self.text_encoder_hidden_size
        img_vae_dim = self.num_channels_latents * latent_height * latent_width
        label_dim = self.num_channels_latents * latent_height * latent_width

        text, img_vae, img_clip, label_clip = x.split([text_dim, img_vae_dim, self.image_encoder_projection_dim, label_dim], dim=1)

        text = torch.reshape(text, (batch_size, self.text_encoder_seq_len, self.text_encoder_hidden_size))
        img_vae = torch.reshape(img_vae, (batch_size, self.num_channels_latents, latent_height, latent_width))
        img_clip = torch.reshape(img_clip, (batch_size, 1, self.image_encoder_projection_dim))
        label = torch.reshape(label_clip, (batch_size, self.num_channels_latents, latent_height, latent_width))
        return text, img_vae, img_clip, label

    @staticmethod
    def _combine_joint(text, img_vae, img_clip, label):
        r"""
        Combines a latent image img_vae of shape (B, C, H, W), a CLIP-embedded image img_clip of shape (B, L_img,
        clip_img_dim), and a text embedding text of shape (B, L_text, text_dim) into a single embedding x of shape (B,
        C * H * W + L_img * clip_img_dim + L_text * text_dim).
        """
        text = torch.reshape(text, (text.shape[0], -1))
        img_vae = torch.reshape(img_vae, (img_vae.shape[0], -1))
        img_clip = torch.reshape(img_clip, (img_clip.shape[0], -1))
        label = torch.reshape(label, (label.shape[0], -1))
        return torch.concat([text, img_vae, img_clip, label], dim=-1)

    def _get_noise_pred(
        self,
        mode,
        latents,
        t,
        text,
        max_timestep,
        data_type,
        guidance_scale,
        generator,
        device,
        height,
        width,
    ):
        r"""
        Gets the noise prediction using the `unet` and performs classifier-free guidance, if necessary.
        """
        if mode == "text2img":
            # Text-conditioned image-label generation
            img_vae_latents, img_clip_embeds, label_latents = self._split(latents, height, width)

            text_out, img_vae_out, img_clip_out, label_out = self.unet(
                text, img_vae_latents, img_clip_embeds, label_latents, timestep_text=0, timestep_img=t, data_type=data_type
            )

            x_out = self._combine(img_vae_out, img_clip_out, label_out)

            if guidance_scale <= 1.0:
                return x_out

            # Classifier-free guidance
            text_t = randn_tensor(text.shape, generator=generator, device=device, dtype=text.dtype)

            _, img_vae_out_uncond, img_clip_out_uncond, label_out_uncond = self.unet(
                text_t,
                img_vae_latents,
                img_clip_embeds,
                label_latents,
                timestep_text=max_timestep,
                timestep_img=t,
                data_type=data_type
            )

            x_out_uncond = self._combine(img_vae_out_uncond, img_clip_out_uncond, label_out_uncond)

            return guidance_scale * x_out + (1.0 - guidance_scale) * x_out_uncond
        else:
            text_embeds, image_vae_latents, image_clip_embeds, label_latents = self._split_joint(latents, height, width)
            text_out, img_vae_out, img_clip_out, label_out = self.unet(
                text_embeds, image_vae_latents, image_clip_embeds, label_latents, timestep_text=t, timestep_img=t, data_type=data_type
            )

            x_out = self._combine_joint(text_out, img_vae_out, img_clip_out, label_out)

            if guidance_scale <= 1.0:
                return x_out

            text_t = randn_tensor(text_embeds.shape, generator=generator, device=device, dtype=text_embeds.dtype)
            image_vae_t = randn_tensor(image_vae_latents.shape, generator=generator, device=device, dtype=image_vae_latents.dtype)
            image_clip_t = randn_tensor(image_clip_embeds.shape, generator=generator, device=device, dtype=image_clip_embeds.dtype)
            label_t = randn_tensor(label_latents.shape, generator=generator, device=device, dtype=label_latents.dtype)

            _, img_vae_out_uncond, img_clip_out_uncond, label_out_uncond = self.unet(
                text_t, image_vae_latents, image_clip_embeds, label_latents, timestep_text=max_timestep, timestep_img=t, data_type=data_type
            )

            text_out_uncond, _, _, _ = self.unet(
                text_embeds, image_vae_t, image_clip_t, label_t, timestep_text=t, timestep_img=max_timestep, data_type=data_type
            )

            x_out_uncond = self._combine_joint(text_out_uncond, img_vae_out_uncond, img_clip_out_uncond, label_out_uncond)

            return guidance_scale * x_out + (1.0 - guidance_scale) * x_out_uncond

    @staticmethod
    def check_latents_shape(latents_name, latents, expected_shape):
        latents_shape = latents.shape
        expected_num_dims = len(expected_shape) + 1  # expected dimensions plus the batch dimension
        expected_shape_str = ", ".join(str(dim) for dim in expected_shape)
        if len(latents_shape) != expected_num_dims:
            raise ValueError(
                f"`{latents_name}` should have shape (batch_size, {expected_shape_str}), but the current shape"
                f" {latents_shape} has {len(latents_shape)} dimensions."
            )
        for i in range(1, expected_num_dims):
            if latents_shape[i] != expected_shape[i - 1]:
                raise ValueError(
                    f"`{latents_name}` should have shape (batch_size, {expected_shape_str}), but the current shape"
                    f" {latents_shape} has {latents_shape[i]} != {expected_shape[i - 1]} at dimension {i}."
                )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        latents,
        prompt_embeds,
        vae_latents,
        clip_embeds,
        label_latents,
        negative_prompt_embeds,
    ):
        # Check inputs before running the generative process.
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # Check provided latents
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        if latents is not None:
            individual_latents_available = (
                prompt_embeds is not None or vae_latents is not None or clip_embeds is not None or label_latents is not None
            )
            if individual_latents_available:
                logger.warning(
                    "You have supplied both `latents` and at least one of `prompt_latents`, `vae_latents`, and"
                    " `clip_latents`. The value of `latents` will override the value of any individually supplied latents."
                )
            # Check shape of full latents
            img_vae_dim = self.num_channels_latents * latent_height * latent_width
            text_dim = self.text_encoder_seq_len * self.text_encoder_hidden_size
            latents_dim = img_vae_dim + self.image_encoder_projection_dim + text_dim
            latents_expected_shape = (latents_dim,)
            self.check_latents_shape("latents", latents, latents_expected_shape)

        # Check individual latent shapes, if present
        if prompt_embeds is not None:
            prompt_embeds_expected_shape = (self.text_encoder_seq_len, self.text_encoder_hidden_size)
            self.check_latents_shape("prompt_embeds", prompt_embeds, prompt_embeds_expected_shape)

        if vae_latents is not None:
            vae_latents_expected_shape = (self.num_channels_latents, latent_height, latent_width)
            self.check_latents_shape("vae_latents", vae_latents, vae_latents_expected_shape)

        if clip_embeds is not None:
            clip_latents_expected_shape = (1, self.image_encoder_projection_dim)
            self.check_latents_shape("clip_embeds", clip_embeds, clip_latents_expected_shape)

        if label_latents is not None:
            label_latents_expected_shape = (self.num_channels_latents, latent_height, latent_width)
            self.check_latents_shape("label_latents", label_latents, label_latents_expected_shape)

        if prompt_embeds is not None and vae_latents is not None and clip_embeds is not None and label_latents is not None:
            if prompt_embeds.shape[0] != vae_latents.shape[0] or prompt_embeds.shape[0] != clip_embeds.shape[0] or prompt_embeds.shape[0] != label_latents.shape[0]:
                raise ValueError(
                    f"All of `prompt_latents`, `vae_latents`, and `clip_latents` are supplied, but their batch"
                    f" dimensions are not equal: {prompt_embeds.shape[0]} != {vae_latents.shape[0]}"
                    f" != {clip_embeds.shape[0]} != {label_latents.shape[0]}."
                )

    @torch.no_grad()
    def __call__(
        self,
        mode: str = "text2img",
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        data_type: Optional[int] = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        ignore_label: int = 0,
        use_color_map: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        vae_latents: Optional[torch.Tensor] = None,
        clip_embeds: Optional[torch.Tensor] = None,
        label_latents: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            data_type (`int`, *optional*, defaults to 1):
                The data type (either 0 or 1). Only used if you are loading a checkpoint which supports a data type
                embedding; this is added for compatibility with the
                [UniDiffuser-v1](https://huggingface.co/thu-ml/unidiffuser-v1) checkpoint.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 8.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for joint
                image-text generation. Can be used to tweak the same generation with different prompts. If not
                provided, a latents tensor is generated by sampling using the supplied random `generator`. This assumes
                a full set of VAE, CLIP, and text latents, if supplied, overrides the value of `prompt_latents`,
                `vae_latents`, and `clip_latents`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for text
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            vae_latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            clip_embeds (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.ImageTextPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Returns:
            [`~pipelines.unidiffuser.ImageTextPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.unidiffuser.ImageTextPipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images and the second element
                is a list of generated texts.
        """

        # 0. Default height and width to unet
        height = height or self.unet_resolution * self.vae_scale_factor
        width = width or self.unet_resolution * self.vae_scale_factor

        # 1. Check inputs
        # Recalculate mode for each call to the pipeline.
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            latents,
            prompt_embeds,
            vae_latents,
            clip_embeds,
            label_latents,
            negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt, if available; otherwise prepare text latents
        if latents is not None:
            prompt_embeds, vae_latents, clip_embeds, label_latents = self._split_joint(latents, height, width)
        # 3.2. Prepare text latent variables, if input not available
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Encode image, if available; otherwise prepare image latents
        # 4.2. Prepare image latent variables, if input not available
        # Prepare image VAE latents in latent space
        image_vae_latents = self.prepare_image_vae_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=vae_latents,
        )

        # Prepare image CLIP latents
        image_clip_embeds = self.prepare_image_clip_embeds(
            batch_size=batch_size * num_images_per_prompt,
            clip_img_dim=self.image_encoder_projection_dim,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=clip_embeds,
        )

        label_embeds = self.prepare_label_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=vae_latents,
        )

        # 5. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # max_timestep = timesteps[0]
        max_timestep = self.scheduler.config.num_train_timesteps

        # 6. Prepare latent variables
        if mode == "text2img":
            latents = self._combine(image_vae_latents, image_clip_embeds, label_embeds)
        else:
            assert mode == "joint", f"Invalid mode: {mode}. Choose between 'text2img' and 'joint'."
            latents = self._combine_joint(prompt_embeds, image_vae_latents, image_clip_embeds, label_embeds)

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        logger.debug(f"Scheduler extra step kwargs: {extra_step_kwargs}")

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # predict the noise residual
                # Also applies classifier-free guidance as described in the UniDiffuser paper
                noise_pred = self._get_noise_pred(
                    mode,
                    latents,
                    t,
                    prompt_embeds,
                    max_timestep,
                    data_type,
                    guidance_scale,
                    generator,
                    device,
                    height,
                    width,
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 9. Post-processing
        if mode == "text2img":
            image_vae_latents, image_clip_embeds, label_latents = self._split(latents, height, width)
        else:
            text_embeds, image_vae_latents, image_clip_embeds, label_latents = self._split_joint(latents, height, width)

        if not output_type == "latent":
            # Map latent VAE image back to pixel space
            image = self.image_vae.decode(image_vae_latents / self.image_vae.config.scaling_factor, return_dict=False)[0]
            label = self.label_vae.decode(label_latents / self.label_vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = image_vae_latents
            label = label_latents

        self.maybe_free_model_hooks()

        # 10. Postprocess the image, if necessary
        if image is not None:
            do_denormalize = [True] * image.shape[0]
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if label is not None:
            predictions = torch.argmax(label, dim=1)
            probs = torch.nn.functional.softmax(label, dim=1)
            probs = torch.max(probs, dim=1).values
            predictions[probs < 0.5] = ignore_label
            predictions = predictions.cpu().numpy()
            if use_color_map:
                label = encode_seg(predictions).astype(np.uint8)
            else:
                label = predictions.astype(np.uint8)
            label_list = []
            for i in range(label.shape[0]):
                label_list.append(Image.fromarray(label[i]))
            label = label_list

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return image, label

        return ImageLabelPipelineOutput(images=image, labels=label)
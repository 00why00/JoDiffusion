from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.pipelines.unidiffuser.modeling_uvit import PatchEmbed, trunc_normal_, UTransformer2DModel
from diffusers.utils import logging
from torch import nn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class JoDiffusionModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        text_dim: int = 768,
        inner_text_dim: int = 64,
        clip_img_dim: int = 512,
        num_text_tokens: int = 77,
        num_attention_heads: int = 24,
        attention_head_dim: int = 64,
        in_channels: int = 4,
        out_channels: int = 4,
        num_layers: int = 30,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: int = 64,
        num_vector_embeds: Optional[int] = None,
        patch_size: int = 2,
        activation_fn: str = "gelu",
        num_embeds_ada_norm: int = 1000,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        block_type: str = "unidiffuser",
        pre_layer_norm: bool = False,
        use_timestep_embedding: bool = False,
        norm_elementwise_affine: bool = True,
        use_patch_pos_embed: bool = False,
        ff_final_dropout: bool = True,
        use_data_type_embedding: bool = True,
    ):
        super().__init__()

        # 0. Handle dimensions
        self.inner_dim = num_attention_heads * attention_head_dim

        assert sample_size is not None, "UniDiffuserModel over patched input must provide sample_size"
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.patch_size = patch_size
        # Assume image is square...
        self.num_patches = (self.sample_size // patch_size) * (self.sample_size // patch_size)

        # 1. Define input layers
        # 1.1 Input layers for label and image input
        # For now, only support patch input for VAE latent image input
        self.pre_text = nn.Linear(text_dim, inner_text_dim)
        self.text_in = nn.Linear(inner_text_dim, self.inner_dim)
        self.vae_img_in = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            use_pos_embed=use_patch_pos_embed,
        )
        self.clip_img_in = nn.Linear(clip_img_dim, self.inner_dim)
        self.vae_label_in = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            use_pos_embed=use_patch_pos_embed,
        )

        # 1.2. Timestep embeddings for t_img, t_label
        self.timestep_img_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_img_embed = (
            TimestepEmbedding(
                self.inner_dim,
                4 * self.inner_dim,
                out_dim=self.inner_dim,
            )
            if use_timestep_embedding
            else nn.Identity()
        )

        self.timestep_text_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_text_embed = (
            TimestepEmbedding(
                self.inner_dim,
                4 * self.inner_dim,
                out_dim=self.inner_dim,
            )
            if use_timestep_embedding
            else nn.Identity()
        )

        # 1.3. Positional embedding
        self.num_text_tokens = num_text_tokens
        self.num_tokens = 1 + 1 + num_text_tokens + self.num_patches + 1 + self.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.inner_dim))
        self.pos_embed_drop = nn.Dropout(p=dropout)
        trunc_normal_(self.pos_embed, std=0.02)

        # 1.4. Handle data type token embeddings for UniDiffuser-V1, if necessary
        self.use_data_type_embedding = use_data_type_embedding
        if self.use_data_type_embedding:
            self.data_type_token_embedding = nn.Embedding(2, self.inner_dim)
            self.data_type_pos_embed_token = nn.Parameter(torch.zeros(1, 1, self.inner_dim))

        # 2. Define transformer blocks
        self.transformer = UTransformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size=sample_size,
            num_vector_embeds=num_vector_embeds,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            block_type=block_type,
            pre_layer_norm=pre_layer_norm,
            norm_elementwise_affine=norm_elementwise_affine,
            use_patch_pos_embed=use_patch_pos_embed,
            ff_final_dropout=ff_final_dropout,
        )

        # 3. Define output layers
        patch_dim = (patch_size ** 2) * out_channels
        self.text_out = nn.Linear(self.inner_dim, inner_text_dim)
        self.post_text = nn.Linear(inner_text_dim, text_dim)
        self.vae_img_out = nn.Linear(self.inner_dim, patch_dim)
        self.clip_img_out = nn.Linear(self.inner_dim, clip_img_dim)
        self.label_out = nn.Linear(self.inner_dim, patch_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        image_latents: torch.Tensor,
        image_embeds: torch.Tensor,
        label_latents: torch.Tensor,
        timestep_img: Union[torch.Tensor, float, int],
        timestep_text: Union[torch.Tensor, float, int],
        data_type: Optional[Union[torch.Tensor, float, int]] = 1,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
    ):
        """
        Args:
            prompt_embeds (`torch.Tensor` of shape `(batch size, seq_len, text_dim)`):
                CLIP-embedded text representation.
            image_latents (`torch.Tensor` of shape `(batch size, latent channels, height, width)`):
                Latent image representation from the VAE encoder.
            image_embeds (`torch.Tensor` of shape `(batch size, 1, clip_img_dim)`):
                CLIP-embedded image representation (unsqueezed in the first dimension).
            label_latents (`torch.Tensor` of shape `(batch size, seq_len, label_dim)`):
                Task-specific-model-embedded label representation.
            timestep_text (`torch.long` or `float` or `int`):
                Current denoising step for the label.
            timestep_img (`torch.long` or `float` or `int`):
                Current denoising step for the image.
            data_type: (`torch.int` or `float` or `int`, *optional*, defaults to `1`):
                Only used in UniDiffuser-v1-style models. Can be either `1`, to use weights trained on nonpublic data,
                or `0` otherwise.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            cross_attention_kwargs (*optional*):
                Keyword arguments to supply to the cross attention layers, if used.


        Returns:
            `tuple`: Returns relevant parts of the model's noise prediction: the first element of the tuple is tbe VAE
            image embedding, the second element is the CLIP image embedding, and the third element is label embedding.
        """
        batch_size = image_latents.shape[0]

        # 1. Input
        # 1.1. Map inputs to shape (B, N, inner_dim)
        text_hidden_states = self.text_in(self.pre_text(prompt_embeds))
        vae_hidden_states = self.vae_img_in(image_latents)
        clip_hidden_states = self.clip_img_in(image_embeds)
        label_hidden_states = self.vae_label_in(label_latents)

        num_text_tokens, num_img_tokens = text_hidden_states.size(1), vae_hidden_states.size(1)

        # 1.2. Encode image timesteps to single token (B, 1, inner_dim)
        if not torch.is_tensor(timestep_img):
            timestep_img = torch.tensor([timestep_img], dtype=torch.long, device=vae_hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep_img = timestep_img * torch.ones(batch_size, dtype=timestep_img.dtype, device=timestep_img.device)

        timestep_img_token = self.timestep_img_proj(timestep_img)
        # t_img_token does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        timestep_img_token = timestep_img_token.to(dtype=self.dtype)
        timestep_img_token = self.timestep_img_embed(timestep_img_token)
        timestep_img_token = timestep_img_token.unsqueeze(dim=1)

        # 1.3. Encode label timesteps to single token (B, 1, inner_dim)
        if not torch.is_tensor(timestep_text):
            timestep_text = torch.tensor([timestep_text], dtype=torch.long, device=vae_hidden_states.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep_text = timestep_text * torch.ones(batch_size, dtype=timestep_text.dtype, device=timestep_text.device)

        timestep_text_token = self.timestep_text_proj(timestep_text)
        # t_label_token does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        timestep_text_token = timestep_text_token.to(dtype=self.dtype)
        timestep_text_token = self.timestep_text_embed(timestep_text_token)
        timestep_text_token = timestep_text_token.unsqueeze(dim=1)

        # 1.4. Concatenate all the embeddings together.
        if self.use_data_type_embedding:
            assert data_type is not None, "data_type must be supplied if the model uses a data type embedding"
            if not torch.is_tensor(data_type):
                data_type = torch.tensor([data_type], dtype=torch.int, device=vae_hidden_states.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            data_type = data_type * torch.ones(batch_size, dtype=data_type.dtype, device=data_type.device)

            data_type_token = self.data_type_token_embedding(data_type).unsqueeze(dim=1)
            hidden_states = torch.cat(
                [
                    timestep_img_token,
                    timestep_text_token,
                    data_type_token,
                    text_hidden_states,
                    clip_hidden_states,
                    vae_hidden_states,
                    label_hidden_states,
                ],
                dim=1,
            )
        else:
            hidden_states = torch.cat(
                [
                    timestep_img_token,
                    timestep_text_token,
                    text_hidden_states,
                    clip_hidden_states,
                    vae_hidden_states,
                    label_hidden_states,
                ],
                dim=1,
            )

        # 1.5. Prepare the positional embeddings and add to hidden states
        # Note: I think img_vae should always have the proper shape, so there's no need to interpolate
        # the position embeddings.
        if self.use_data_type_embedding:
            pos_embed = torch.cat(
                [self.pos_embed[:, : 1 + 1, :], self.data_type_pos_embed_token, self.pos_embed[:, 1 + 1:, :]], dim=1
            )
        else:
            pos_embed = self.pos_embed
        hidden_states = hidden_states + pos_embed
        hidden_states = self.pos_embed_drop(hidden_states)

        # 2. Blocks
        hidden_states = self.transformer(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=None,
            class_labels=None,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            hidden_states_is_embedding=True,
            unpatchify=False,
        )[0]

        # 3. Output
        # Split out the predicted noise representation.
        if self.use_data_type_embedding:
            (
                t_img_token_out,
                t_text_token_out,
                data_type_token_out,
                text_out,
                img_clip_out,
                img_vae_out,
                label_out,
            ) = hidden_states.split((1, 1, 1, num_text_tokens, 1, num_img_tokens, num_img_tokens), dim=1)
        else:
            t_img_token_out, t_text_token_out, text_out, img_clip_out, img_vae_out, label_out = hidden_states.split(
                (1, 1, num_text_tokens, 1, num_img_tokens, num_img_tokens), dim=1
            )

        text_out = self.post_text(self.text_out(text_out))

        img_clip_out = self.clip_img_out(img_clip_out)

        img_vae_out = self.vae_img_out(img_vae_out)
        height = width = int(img_vae_out.shape[1] ** 0.5)
        img_vae_out = img_vae_out.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        img_vae_out = torch.einsum("nhwpqc->nchpwq", img_vae_out)
        img_vae_out = img_vae_out.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        label_out = self.label_out(label_out)
        label_out = label_out.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        label_out = torch.einsum("nhwpqc->nchpwq", label_out)
        label_out = label_out.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        return text_out, img_vae_out, img_clip_out, label_out
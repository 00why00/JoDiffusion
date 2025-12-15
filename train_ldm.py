import argparse
import contextlib
import functools
import gc
import logging
import os
import random
import shutil
from datetime import timedelta

import datasets
import diffusers
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, concatenate_datasets
from datasets.fingerprint import Hasher
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from pipelines.pipeline_jodiffusion import JoDiffusionPipeline
from pipelines.modeling_uvit import JoDiffusionModel
from train_ae import Mask2Tensor, EncodeBitMap

logger = get_logger(__name__, log_level="INFO")


def log_validation(unet, args, accelerator, weight_dtype, epoch, is_final_validation=False):
    logger.info("Running validation... ")
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    if not is_final_validation:
        unet = accelerator.unwrap_model(unet)
    else:
        unet = JoDiffusionModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    pipeline_kwargs = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "unet": unet, "torch_dtype": weight_dtype,
    }
    if args.lightweight_label_vae:
        from pipelines.modeling_lightweight_vae import LightweightLabelVAE
        pipeline_kwargs["label_vae"] = LightweightLabelVAE.from_pretrained(args.pretrained_label_vae_path, torch_dtype=weight_dtype)
    else:
        pipeline_kwargs["label_vae"] = AutoencoderKL.from_pretrained(args.pretrained_label_vae_path, torch_dtype=weight_dtype)
    pipeline = JoDiffusionPipeline.from_pretrained(**pipeline_kwargs)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    if args.dataset_name == "ade20k_semantic":
        prompt = "a bathroom with a toilet and a shower."
    elif args.dataset_name == "coco_semantic":
        prompt = "a person riding a bike on a street with a car parked on the side of the road."
    elif args.dataset_name == "voc_semantic":
        prompt = "a person walking a dog on a leash with a car parked on the side of the road."
    else:
        raise ValueError(f"Unknown dataset {args.dataset_name}")

    images = []
    labels = []
    for iii in range(4):
        with inference_ctx:
            sample = pipeline(
                mode="text2img" if iii < 2 else "joint",
                prompt=prompt,
                num_inference_steps=50,
                generator=generator,
                ignore_label=0,
            )
            image = sample.images[0]
            label = sample.labels[0]
            images.append(image)
            labels.append(label)

    tracker_key = "test" if is_final_validation else f"validation-epoch{epoch}"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            table_data = []

            for idx in range(len(images)):
                table_data.append([
                    wandb.Image(images[idx], caption="Generated Image"),
                    wandb.Image(labels[idx], caption="Generated Label"),
                ])
            tracker.log({tracker_key: wandb.Table(data=table_data, columns=["Image", "Label"])})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="training script of jodiffusion.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='inference/saved_pipeline/jodiffusion',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_label_vae_path",
        type=str,
        default=None,
        help="Path to pretrained label vae model.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ade20k_semantic",
        help="The name of the dataset to use for training.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="blip2",
        choices=["category_name", "blip2", "coco"],
        help="The name of the column in the dataset that contains the captions.",
    )
    parser.add_argument(
        "--lightweight_label_vae",
        action="store_true",
        help="Whether or not to use lightweight vae.",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="image_only",
        choices=["image_only", "joint"],
        help="The type of noise to use for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this"
        " resolution",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop the input images to the resolution. If not set, the images will be randomly"
        " cropped. The images will be resized to the resolution first before cropping.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
        " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=lambda x:None if x == 'None' else str(x), help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="jodiffusion-semantic",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if "debug" in args.output_dir:
        os.environ["WANDB_MODE"] = "offline"

    return args


def encode_prompt(batch, text_encoder, clip_tokenizer, caption_column):
    prompt = batch[caption_column]
    if isinstance(prompt[0], list):
        prompt = [p[0] for p in prompt]
    with torch.no_grad():
        text_inputs = clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            input_ids.to(text_encoder.device),
        )[0]
    return {"prompt_embeds": prompt_embeds.cpu()}


def encode_images(batch, image_vae, image_encoder, clip_image_processor, label_vae, image_column):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(image_vae.device, dtype=image_vae.dtype)

    labels = batch.pop("label_pixel_values")
    label_pixel_values = torch.stack(list(labels))
    label_pixel_values = label_pixel_values.to(memory_format=torch.contiguous_format).float()
    label_pixel_values = label_pixel_values.to(label_vae.device, dtype=label_vae.dtype)

    clip_image = clip_image_processor.preprocess(batch[image_column], return_tensors="pt").data['pixel_values']
    clip_image = torch.stack(list(clip_image))
    clip_image = clip_image.to(memory_format=torch.contiguous_format).float()
    clip_image = clip_image.to(image_encoder.device, dtype=image_encoder.dtype)

    with (torch.no_grad()):
        image_latents = image_vae.encode(pixel_values).latent_dist.sample()
        image_latents = image_latents * image_vae.config.scaling_factor

        label_latents = label_vae.encode(label_pixel_values).latent_dist.sample()
        label_latents = label_latents * label_vae.config.scaling_factor

        image_embeds = image_encoder(clip_image).image_embeds

    return {"image_latents": image_latents.cpu(), "label_latents": label_latents.cpu(), "image_embeds": image_embeds.cpu()}


def make_train_dataset(args, accelerator):
    dataset = load_dataset(f"../dataset/{args.dataset_name}.py", trust_remote_code=True)
    if args.dataset_name == "ade20k_semantic":
        from dataset.ade20k_semantic import ADE20KSemantic
        image_column = "image"
        label_column = "semantic"
        dataset_cls = ADE20KSemantic()
    elif args.dataset_name == "coco_semantic":
        from dataset.coco_semantic import COCOSemantic
        image_column = "image"
        label_column = "semantic"
        dataset_cls = COCOSemantic()
    elif args.dataset_name == "voc_semantic":
        from dataset.voc_semantic import VOCSemantic
        image_column = "image"
        label_column = "semantic"
        dataset_cls = VOCSemantic()
    else:
        raise ValueError(f"Unknown dataset {args.dataset_name}")

    # Preprocessing the datasets.
    img_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    lbl_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST)
    crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
    flip = transforms.RandomHorizontalFlip(p=1.0)
    image2tensor = transforms.ToTensor()
    label2tensor = Mask2Tensor()
    image_norm = transforms.Normalize([0.5], [0.5])
    label_norm = EncodeBitMap(n=math.ceil(math.log(dataset_cls.num_classes, 2)), ignore_label=dataset_cls.ignore_label)

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        labels = [label for label in examples[label_column]]
        pixel_values = []
        label_pixel_values = []
        for image, label in zip(images, labels):
            # resize
            image = img_resize(image)
            label = lbl_resize(label)
            # crop
            image = crop(image)
            label = crop(label)
            # flip
            if args.random_flip and random.random() < 0.5:
                image = flip(image)
                label = flip(label)
            # to tensor
            image = image2tensor(image)
            label = label2tensor(label)
            # norm
            image = image_norm(image)
            label = label_norm(label)[0]
            label = 2. * label - 1.

            pixel_values.append(image)
            label_pixel_values.append(label)

        examples["pixel_values"] = pixel_values
        examples["label_pixel_values"] = label_pixel_values
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset, image_column, dataset_cls


def collate_fn(examples):
    _prompt_embeds = torch.stack([example["prompt_embeds"] for example in examples])
    _image_latents = torch.stack([example["image_latents"] for example in examples])
    _label_latents = torch.stack([example["label_latents"] for example in examples])
    _image_embeds = torch.stack([example["image_embeds"] for example in examples])
    return {
        "prompt_embeds": _prompt_embeds,
        "image_latents": _image_latents,
        "label_latents": _label_latents,
        "image_embeds": _image_embeds,
    }


def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(hours=20))
        ]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_vae")
    if args.lightweight_label_vae:
        from pipelines.modeling_lightweight_vae import LightweightLabelVAE
        label_vae = LightweightLabelVAE.from_pretrained(args.pretrained_label_vae_path)
    else:
        label_vae = AutoencoderKL.from_pretrained(args.pretrained_label_vae_path)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="clip_tokenizer")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="clip_image_processor")
    unet = JoDiffusionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze vae and text_encoder and set unet to trainable
    image_vae.requires_grad_(False)
    label_vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.train()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # The VAE is in float32 to avoid NaN losses.
    image_vae.to(accelerator.device, dtype=torch.float32)
    label_vae.to(accelerator.device, dtype=torch.float32)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for _, model in enumerate(models):
                if isinstance(unwrap_model(model), JoDiffusionModel):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                else:
                    raise ValueError(f"Unknown model type {model}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(model, JoDiffusionModel):
                load_model = JoDiffusionModel.from_pretrained(input_dir, subfolder="unet")
            else:
                raise ValueError(f"Unknown model type {model}")

            model.register_to_config(**load_model.config)  # noqa
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset, image_column, dataset_cls = make_train_dataset(args, accelerator)

    # Precompute the prompt embeddings and image embeddings
    compute_prompt_embeds_fn = functools.partial(
        encode_prompt,
        text_encoder=text_encoder,
        clip_tokenizer=clip_tokenizer,
        caption_column=args.caption_column + "_caption"
    )
    compute_image_embeds_fn = functools.partial(
        encode_images,
        image_vae=image_vae,
        image_encoder=image_encoder,
        clip_image_processor=clip_image_processor,
        label_vae=label_vae,
        image_column=image_column,
    )

    with accelerator.main_process_first():
        fingerprint_for_text = Hasher.hash(
            args.dataset_name +
            "text" +
            args.caption_column
        )
        fingerprint_for_image = Hasher.hash(
            args.dataset_name +
            "image" +
            ("woflip" if not args.random_flip else "") +
            args.pretrained_label_vae_path
        )
        train_dataset_with_prompt_embeds = train_dataset.map(
            compute_prompt_embeds_fn,
            batched=True,
            batch_size=args.train_batch_size * accelerator.num_processes,
            new_fingerprint=fingerprint_for_text,
        )
        train_dataset_with_image_embeds = train_dataset.map(
            compute_image_embeds_fn,
            batched=True,
            batch_size=args.train_batch_size * accelerator.num_processes,
            new_fingerprint=fingerprint_for_image,
        )
        duplicated_columns = list(set(train_dataset_with_prompt_embeds.column_names).intersection(train_dataset_with_image_embeds.column_names))
        logger.info(f"Duplicated columns: {duplicated_columns}")
        precomputed_dataset = concatenate_datasets(
            [train_dataset_with_prompt_embeds, train_dataset_with_image_embeds.remove_columns(duplicated_columns)], axis=1
        ).with_format('torch')

    del compute_prompt_embeds_fn, compute_image_embeds_fn
    del text_encoder, clip_tokenizer, image_encoder, clip_image_processor, image_vae, label_vae
    gc.collect()
    torch.cuda.empty_cache()

    # DataLoaders creation:
    train_dataloader = DataLoader(
        precomputed_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler  # noqa
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(precomputed_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))  # noqa
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            loss = 0.0
            with accelerator.accumulate(unet):
                text_embeds = batch["prompt_embeds"].to(accelerator.device)
                text_embeds_shape = text_embeds.shape
                text_embeds = torch.reshape(text_embeds, (text_embeds.shape[0], -1))

                image_vae_latents = batch["image_latents"].to(accelerator.device)
                image_vae_shape = image_vae_latents.shape
                image_vae_latents = torch.reshape(image_vae_latents, (image_vae_latents.shape[0], -1))

                image_clip_embeds = batch["image_embeds"].unsqueeze(1).to(accelerator.device)
                image_clip_shape = image_clip_embeds.shape
                image_clip_embeds = torch.reshape(image_clip_embeds, (image_clip_embeds.shape[0], -1))

                label_latents = batch["label_latents"].to(accelerator.device)
                label_shape = label_latents.shape
                label_latents = torch.reshape(label_latents, (label_latents.shape[0], -1))

                image_feature = torch.cat([image_vae_latents, image_clip_embeds, label_latents], dim=1)

                if args.noise_type == "image_only":
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(image_feature)
                    bsz = image_feature.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=image_feature.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(image_feature, noise, timesteps)

                    input_text_embeds = torch.reshape(text_embeds, text_embeds_shape)
                    noised_image_vae_latents, noised_image_clip_embeds, noised_label_latents = torch.split(
                        noisy_latents, [image_vae_latents.shape[-1], image_clip_embeds.shape[-1], label_latents.shape[-1]], dim=1)
                    noised_image_vae_latents = torch.reshape(noised_image_vae_latents, image_vae_shape)
                    noised_image_clip_embeds = torch.reshape(noised_image_clip_embeds, image_clip_shape)
                    noised_label_latents = torch.reshape(noised_label_latents, label_shape)
                elif args.noise_type == "joint":
                    # Sample noise that we'll add to the latents
                    text_noise = torch.randn_like(text_embeds)
                    image_noise = torch.randn_like(image_feature)
                    bsz = image_feature.shape[0]
                    # Sample a random timestep for each image
                    text_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=text_embeds.device)
                    image_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=image_feature.device)
                    text_timesteps = text_timesteps.long()
                    image_timesteps = image_timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    text_noisy_latents = noise_scheduler.add_noise(text_embeds, text_noise, text_timesteps)
                    image_noisy_latents = noise_scheduler.add_noise(image_feature, image_noise, image_timesteps)

                    input_text_embeds = torch.reshape(text_noisy_latents, text_embeds_shape)
                    noised_image_vae_latents, noised_image_clip_embeds, noised_label_latents = torch.split(
                        image_noisy_latents,
                        [image_vae_latents.shape[-1], image_clip_embeds.shape[-1], label_latents.shape[-1]], dim=1)
                    noised_image_vae_latents = torch.reshape(noised_image_vae_latents, image_vae_shape)
                    noised_image_clip_embeds = torch.reshape(noised_image_clip_embeds, image_clip_shape)
                    noised_label_latents = torch.reshape(noised_label_latents, label_shape)
                else:
                    raise ValueError(f"Unknown noise type {args.noise_type}")

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    if args.noise_type == "image_only":
                        target = noise
                    elif args.noise_type == "joint":
                        target = torch.cat([text_noise, image_noise], dim=1)
                    else:
                        raise ValueError(f"Unknown noise type {args.noise_type}")
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                text_out, image_vae_out, image_clip_out, label_out = unet(
                    input_text_embeds,
                    noised_image_vae_latents,
                    noised_image_clip_embeds,
                    noised_label_latents,
                    timestep_text=0 if args.noise_type == "image_only" else text_timesteps,
                    timestep_img=timesteps if args.noise_type == "image_only" else image_timesteps,
                )
                text_out = torch.reshape(text_out, (text_out.shape[0], -1))
                image_vae_out = torch.reshape(image_vae_out, (image_vae_out.shape[0], -1))
                image_clip_out = torch.reshape(image_clip_out, (image_clip_out.shape[0], -1))
                label_out = torch.reshape(label_out, (label_out.shape[0], -1))

                if args.noise_type == "image_only":
                    combine_out = torch.cat([image_vae_out, image_clip_out, label_out], dim=1)
                else:
                    combine_out = torch.cat([text_out, image_vae_out, image_clip_out, label_out], dim=1)

                loss += F.mse_loss(combine_out.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                log_validation(
                    unet,
                    args,
                    accelerator,
                    weight_dtype,
                    epoch,
                )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet.save_pretrained(args.output_dir)

        log_validation(
            unet,
            args,
            accelerator,
            weight_dtype,
            epoch=None,
            is_final_validation=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
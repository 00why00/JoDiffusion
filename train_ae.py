import argparse
import logging
import math
import os
import shutil
from datetime import timedelta

import datasets
import diffusers
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from pipelines.loss_segmentation import SegmentationLosses
from validate.miou_evaluation import MeanIoUEvaluator

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def log_validation(vae, val_dataloader, dataset_cls, args, accelerator, global_step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        vae = accelerator.unwrap_model(vae)
    else:
        if args.lightweight_label_vae:
            from pipelines.modeling_lightweight_vae import LightweightLabelVAE as VAEClass
        else:
            from diffusers import AutoencoderKL as VAEClass
        vae = VAEClass.from_pretrained(args.output_dir)

    vae = vae.to(accelerator.device)
    vae.eval()

    miou_meter = MeanIoUEvaluator(
        num_classes=dataset_cls.num_classes,
        class_names=dataset_cls.category_names,
        has_bg=True,
        ignore_index=255,
    )

    for batch_idx, data in tqdm(enumerate(val_dataloader)):
        label_input = data["bit_pixel_values"].to(accelerator.device)
        label_target = data["pixel_values"].to(accelerator.device)
        label_input = 2. * label_input - 1.

        output = vae(label_input, sample_posterior=False).sample
        mask_th = 0.5

        # compute miou
        pred = torch.argmax(output, dim=1)
        prob = torch.nn.functional.softmax(output, dim=1)
        prob = prob.max(dim=1)[0]
        pred[prob < mask_th] = dataset_cls.ignore_label # set to background if the probability is less than 0.5
        miou_meter.update(pred, label_target)

    miou_results = miou_meter.return_score(verbose=False, name='val set')
    accelerator.log({
        "val/semantic mIoU": miou_results['mIoU'],
        "val/semantic IoU pre class": wandb.Table(data=[miou_results['jaccards_all_categs']], columns=dataset_cls.category_names)
    }, step=global_step)
    return


def parse_args():
    parser = argparse.ArgumentParser(description="training script of dense label vae.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ade20k_semantic",
        help="The name of the dataset to use for training.",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="semantic",
        help="The column name of the label in the dataset.",
    )
    parser.add_argument(
        "--lightweight_label_vae",
        action="store_true",
        help="Whether or not to use lightweight vae.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=['ce', 'seg'],
        help="The type of loss to use for training.",
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
    parser.add_argument("--max_grad_norm", default=3.0, type=float, help="Max gradient norm.")
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
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vae-semantic",
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


class Mask2Tensor(torch.nn.Module):
    def __call__(self, x):
        return torch.from_numpy(np.array(x)).long()

class EncodeBitMap(torch.nn.Module):
    def __init__(self, n=8, fill_value=0.5, ignore_label=0):
        super().__init__()
        self.n = n
        self.fill_value = fill_value
        self.ignore_label = ignore_label

    def __call__(self, x):
        ignore_mask = x == self.ignore_label
        x = torch.bitwise_right_shift(x, torch.arange(self.n, device=x.device)[:, None, None])
        x = torch.remainder(x, 2).float()
        x[:, ignore_mask] = self.fill_value
        return x, ignore_mask

def make_train_dataset(args, accelerator):
    dataset = load_dataset(f"../dataset/{args.dataset_name}.py", trust_remote_code=True)
    if args.dataset_name == "ade20k_semantic":
        from dataset.ade20k_semantic import ADE20KSemantic
        dataset_cls = ADE20KSemantic()
    elif args.dataset_name == "coco_semantic":
        from dataset.coco_semantic import COCOSemantic
        dataset_cls = COCOSemantic()
    elif args.dataset_name == "voc_semantic":
        from dataset.voc_semantic import VOCSemantic
        dataset_cls = VOCSemantic()
    else:
        raise ValueError(f"Unknown dataset name {args.dataset_name}")

    # Preprocessing the datasets.
    train_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        Mask2Tensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(args.resolution),
        Mask2Tensor(),
    ])
    encode = EncodeBitMap(n=math.ceil(math.log(dataset_cls.num_classes, 2)), ignore_label=dataset_cls.ignore_label)

    def preprocess_train(examples):
        labels = examples[args.label_column]
        examples["pixel_values"] = [train_transform(label) for label in labels]
        examples["bit_pixel_values"] = [encode(label)[0] for label in examples["pixel_values"]]
        return examples

    def preprocess_val(examples):
        labels = examples[args.label_column]
        examples["pixel_values"] = [val_transform(label) for label in labels]
        examples["bit_pixel_values"] = [encode(label)[0] for label in examples["pixel_values"]]
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)
        val_dataset = dataset["validation"].with_transform(preprocess_val)

    return train_dataset, val_dataset, dataset_cls


def collate_fn(examples):
    _pixel_values = torch.stack([example["pixel_values"] for example in examples])
    _pixel_values = _pixel_values.to(memory_format=torch.contiguous_format).long()
    _bit_pixel_values = torch.stack([example["bit_pixel_values"] for example in examples])
    _bit_pixel_values = _bit_pixel_values.to(memory_format=torch.contiguous_format).float()
    _meta = [example["meta"] for example in examples]
    return {
        "pixel_values": _pixel_values,
        "bit_pixel_values": _bit_pixel_values,
        "meta": _meta,
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
            DistributedDataParallelKwargs(find_unused_parameters=False),
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

    # Load dataset
    train_dataset, val_dataset, dataset_cls = make_train_dataset(args, accelerator)

    # Load models.
    in_channels = math.ceil(math.log(dataset_cls.num_classes, 2))
    out_channels = dataset_cls.num_classes + 1
    if args.lightweight_label_vae:
        from pipelines.modeling_lightweight_vae import LightweightLabelVAE as VAEClass
        vae = VAEClass(in_channels=in_channels, out_channels=out_channels)
    else:
        from diffusers import AutoencoderKL as VAEClass
        config = VAEClass.load_config("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
        config["in_channels"] = in_channels
        config["out_channels"] = out_channels
        vae = VAEClass(**config)

    logger.info(f"VAE in channels: {in_channels}, out channels: {out_channels}")

    vae.train()
    vae.to(accelerator.device, dtype=torch.float32)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for _, model in enumerate(models):
                if isinstance(unwrap_model(model), VAEClass):
                    model.save_pretrained(os.path.join(output_dir, "vae"))
                else:
                    raise ValueError(f"Unknown model type {model}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(model, VAEClass):
                load_model = VAEClass.from_pretrained(input_dir, subfolder="vae")
            else:
                raise ValueError(f"Unknown model type {model}")

            model.register_to_config(**load_model.config)  # noqa
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

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
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
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
    vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler  # noqa
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "  # noqa
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        for i in range(10):
            try:
                accelerator.init_trackers(args.tracker_project_name, tracker_config)
                break
            except Exception as e:
                logger.error(f"Error initializing trackers: {e}")

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
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
        vae.train()
        seg_loss = SegmentationLosses(ignore_label=255)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(vae):
                loss = 0.0
                label_input = batch["bit_pixel_values"].to(accelerator.device)
                label_target = batch["pixel_values"].to(accelerator.device)
                label_input = 2. * label_input - 1.

                output = vae(label_input, sample_posterior=True).sample

                if args.loss_type == "ce":
                    loss += torch.nn.functional.cross_entropy(output, label_target, reduction="mean", ignore_index=255)
                elif args.loss_type == "seg":
                    seg_ls = seg_loss.point_loss(output, label_target)
                    loss += seg_ls['ce'] * 1.0 + seg_ls['mask'] * 1.0
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train/loss": train_loss,}, step=global_step)
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
                    vae,
                    val_dataloader,
                    dataset_cls,
                    args,
                    accelerator,
                    global_step,
                )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = unwrap_model(vae)
        vae.save_pretrained(args.output_dir)

        log_validation(
            vae,
            val_dataloader,
            dataset_cls,
            args,
            accelerator,
            global_step=global_step,
            is_final_validation=True,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
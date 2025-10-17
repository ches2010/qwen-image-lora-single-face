#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Qwen-Image LoRA 微调脚本（单张人脸，自动检测过拟合）
# 兼容 Qwen/Qwen-Image (DiT-based 20B T2I model)

import argparse
import logging
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, PretrainedConfig
import cv2
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=None,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args():
    parser = argparse.ArgumentParser(description="Single image face training with overfitting detection.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Qwen/Qwen-Image")
    parser.add_argument("--instance_image_path", type=str, default="dataset/001.jpg", help="Path to single instance image")
    parser.add_argument("--concept_prompt_file", type=str, default="concept_prompt.txt", help="File containing instance prompt like 'sks person'")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)  # Fixed epochs for single image
    parser.add_argument("--max_train_steps", type=int, default=600)  # Conservative steps
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)  # Lower rank for single image
    parser.add_argument("--output_dir", type=str, default="qwen-image-lora-single-face")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=100, help="Steps interval for evaluation")
    parser.add_argument("--eval_prompt", type=str, default="a photo of sks person", help="Prompt for evaluation")
    parser.add_argument("--num_eval_images", type=int, default=1, help="Number of eval images to generate per eval")
    parser.add_argument("--ssim_threshold", type=float, default=0.95, help="SSIM threshold to detect overfitting")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    args = parser.parse_args()
    return args


def calculate_ssim(img1_path, img2_path):
    """Calculate SSIM between two images"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0.0
    
    # Resize to same size if needed
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (min_w, min_h))
    img2 = cv2.resize(img2, (min_w, min_h))
    
    score, _ = ssim(img1, img2, full=True)
    return score


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        eval_dir = Path(args.output_dir, "eval_images")
        eval_dir.mkdir(exist_ok=True)

    # Load concept prompt
    with open(args.concept_prompt_file, "r") as f:
        instance_prompt = f.read().strip()
    # Extract class prompt by removing trigger word (e.g. 'sks person' -> 'person')
    trigger_word = instance_prompt.split()[0]  # 'sks'
    class_prompt = instance_prompt.replace(trigger_word, "", 1).strip()  # 'person'
    logger.info(f"Using instance prompt: '{instance_prompt}' and class prompt: '{class_prompt}'")

    # Load original instance image for overfitting detection
    original_img_path = args.instance_image_path
    if not os.path.exists(original_img_path):
        raise FileNotFoundError(f"Instance image not found: {original_img_path}")
    
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze VAE and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Set unet to trainable and add LoRA
    unet.requires_grad_(False)
    unet_lora_config = diffusers.LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,  # Add dropout to prevent overfitting
    )
    unet.add_adapter(unet_lora_config)

    # Enable xformers if available
    if args.enable_xformers_memory_efficient_attention:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except:
            logger.warning("xformers not available or not compatible")

    # Data Augmentation for single image
    img = Image.open(args.instance_image_path).convert("RGB")
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
        transforms.RandomRotation(degrees=5),    # Small rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Minor color jitter
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def generate_augmented_samples(img, num_samples=20):  # Generate more augmented samples
        samples = []
        prompts = []
        for _ in range(num_samples):
            aug_img = image_transforms(img)
            samples.append(aug_img)
            prompts.append(instance_prompt)
        return torch.stack(samples), prompts

    aug_images, aug_prompts = generate_augmented_samples(img, num_samples=20)
    aug_dataset = Dataset.from_dict({
        "pixel_values": [img for img in aug_images],
        "input_ids": tokenizer(
            aug_prompts,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids
    })

    train_dataloader = torch.utils.data.DataLoader(
        aug_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0
    )

    # Optimizer and LR scheduler
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate, weight_decay=1e-2)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    global_step = 0
    unet.train()
    
    # For evaluation
    eval_pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16
    ).to(accelerator.device)
    eval_pipe.unet = accelerator.unwrap_model(unet)  # Use the trained unet
    
    # Variables for overfitting detection
    ssim_scores = []
    best_ssim = 0.0
    patience_counter = 0
    early_stop = False
    
    for epoch in range(args.num_train_epochs):
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(progress_bar):
            if early_stop:
                break
                
            with accelerator.accumulate(unet):
                # Encode images to latent
                latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode prompt
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.set_postfix({"loss": loss.detach().item(), "step": global_step})

                if global_step % args.eval_steps == 0:
                    logger.info(f"Step {global_step}, Loss: {loss.detach().item():.4f}")
                    
                    # Generate evaluation images
                    if accelerator.is_main_process:
                        eval_pipe.load_lora_weights(args.output_dir)  # Load current LoRA weights
                        eval_pipe.set_progress_bar_config(disable=True)
                        
                        for i in range(args.num_eval_images):
                            eval_image = eval_pipe(
                                args.eval_prompt,
                                num_inference_steps=25,
                                guidance_scale=7.5
                            ).images[0]
                            eval_image_path = f"{args.output_dir}/eval_images/eval_step_{global_step}_img_{i}.png"
                            eval_image.save(eval_image_path)
                            
                            # Calculate SSIM between generated image and original
                            ssim_score = calculate_ssim(original_img_path, eval_image_path)
                            ssim_scores.append(ssim_score)
                            
                            logger.info(f"Generated image SSIM with original: {ssim_score:.4f}")
                            
                            # Check for overfitting
                            if ssim_score > args.ssim_threshold:
                                logger.warning(f"SSIM {ssim_score:.4f} exceeds threshold {args.ssim_threshold} - potential overfitting detected!")
                                
                                # Check if SSIM has been consistently high (indicating overfitting)
                                if len(ssim_scores) >= 3:
                                    recent_scores = ssim_scores[-3:]
                                    if all(s > args.ssim_threshold for s in recent_scores):
                                        patience_counter += 1
                                        
                                        if patience_counter >= args.early_stopping_patience:
                                            logger.warning(f"Early stopping triggered at step {global_step} due to overfitting detection")
                                            early_stop = True
                                            break
                                    else:
                                        patience_counter = 0  # Reset counter if not consistently high
                                else:
                                    patience_counter = 0
                            else:
                                # Reset patience counter if SSIM is below threshold
                                patience_counter = 0
                        
                        # Save checkpoint
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                        # Save current LoRA weights temporarily
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unet_lora_state_dict = diffusers.utils.convert_state_dict_to_diffusers(
                            diffusers.utils.get_peft_model_state_dict(unwrapped_unet)
                        )
                        diffusers.loaders.UNet2DConditionLoadersMixin.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                        )
                        
                        # Log overfitting status
                        logger.info(f"Current SSIM: {ssim_score:.4f}, Patience counter: {patience_counter}/{args.early_stopping_patience}")

            if global_step >= args.max_train_steps or early_stop:
                break

    # Final save
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = diffusers.utils.convert_state_dict_to_diffusers(
            diffusers.utils.get_peft_model_state_dict(unet)
        )
        diffusers.loaders.UNet2DConditionLoadersMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
        )
        logger.info(f"Final LoRA weights saved to {args.output_dir}")
        
        if early_stop:
            logger.info(f"Training stopped early at step {global_step} due to overfitting detection")


if __name__ == "__main__":
    main()

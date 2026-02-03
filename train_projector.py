import argparse
import os
from projector.projector import *
from utils import *
from torch.utils.data import DataLoader, Subset
import torch
from tqdm import tqdm
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import csv
from torch.nn.utils import clip_grad_norm_
import subprocess
import sys
import re
try:
    import wandb  # Optional
except ImportError:
    wandb = None





def get_args():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--model_name", type=str, default="llava", choices=["qwen", "llava","llama", "glm4.1v-thinking"], help="model name")
    parser.add_argument("--coco_dir", type=str, default="PathtoCOCO/COCO", help="Path to COCO dataset root")
    parser.add_argument("--cache_path", type=str, default="PathtoCache", help="Path to cache directory for HF models")
    parser.add_argument("--save_projector_dir", type=str, default="./projector/models")

    # Training
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine"], help="LR scheduler type")
    parser.add_argument("--step_size", type=int, default=5, help="StepLR step_size (epochs)")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma")
    parser.add_argument("--cosine_tmax", type=int, default=10, help="CosineAnnealingLR T_max (epochs)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of optimizer steps to linearly warm up the LR")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--context_dim", type=int, default=1024, help="Context dimension for the projector")
    parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension for the projector")
    parser.add_argument("--no_selected_tokens", type=int, default=32, help="Number of tokens to select for training")
    parser.add_argument("--coco_subset", type=float, default=0.1, help="Fraction of COCO dataset to use for training (1.0 means full dataset)")

    # Others
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--log_every_step", action="store_true", help="If set, log metrics at every training/validation step")
    parser.add_argument("--stop_on_nan", action="store_true", help="Stop training when a NaN/Inf loss is encountered")
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="hallucination_attack", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (defaults to log name)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team) name")

    # Test invocation
    parser.add_argument("--run_test_each_epoch", default=False, action="store_true", help="Run test_projector.py after each epoch")
    parser.add_argument("--test_script", type=str, default=os.path.join(os.path.dirname(__file__), "test_projector.py"), help="Path to test_projector.py")
    parser.add_argument("--test_data_path", type=str, default="PathToCOCO", help="Dataset path for test script")
    parser.add_argument("--test_cache_path", type=str, default=None, help="Cache path for test script (defaults to --cache_path)")
    parser.add_argument("--test_target_objects", type=str, default="vase,boat,bird,giraffe,car,remote", help="Comma-separated target objects for test script")

    return parser.parse_args()


def get_name(args):
    return f"{args.model_name}_projector"

def get_log_name(args):
    return f"{args.model_name}_bs={args.batch_size}_lr={args.lr}_epochs={args.epochs}_context_dim={args.context_dim}_hidden_dim={args.hidden_dim}_coco_subset={args.coco_subset}"


@torch.no_grad()
def save_embeddings(loader, model, processor, clip_model, clip_preprocess, output_path):
    all_model, all_clip, all_paths = [], [], []

    for images, paths in tqdm(loader, desc=f"Extracting embeddings"):
        clip_embs = get_clip_image_features(images, clip_model, clip_preprocess)
        llava_embs = get_llava_image_features(images, model, processor)
        all_clip.append(clip_embs)
        all_model.append(llava_embs)
        all_paths.extend(paths)
    
    all_clip_cat = torch.cat(all_clip)
    all_model_cat = torch.cat(all_model)
    logger.info(f"Clip embeddings shape: {all_clip_cat.shape}, Model embeddings shape: {all_model_cat.shape}")
    out = {
        "clip": all_clip_cat,
        "model": all_model_cat,
        "paths": all_paths
    }
    torch.save(out, output_path)

def train_projector(args):
    logger.info("Starting projector training...")
    num_tokens, target_dim = get_num_tokens(args.model_name)
    decoder = TokenMLP(context_dim=args.context_dim, hidden_dim=args.hidden_dim, num_tokens=num_tokens, target_dim=target_dim).cuda()
    logger.info(f"Decoder: {decoder}")

    logger.info(f"Loading {args.model_name}")
    model, processor = get_model(args.model_name, args.cache_path)
    model = model.to("cuda").eval()
    
    logger.info("Loading CLIP Model...")
    clip_model, clip_preprocess = get_clip_model(args.cache_path)
    clip_model.eval().cuda()
    
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = ImageDataset(os.path.join(args.coco_dir, "train2017"), resize=(336, 336))
    indices = random.sample(range(len(train_dataset)), int(args.coco_subset * len(train_dataset)))  # Use subset of the dataset for training
    train_dataset = Subset(train_dataset, indices)
    logger.info(f"Train dataset size: {len(train_dataset)}")
    val_dataset = ImageDataset(os.path.join(args.coco_dir, "val2017"), resize=(336, 336))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x)))

    criterion = nn.MSELoss()   
    
    # Scheduler (optional)
    scheduler = None
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_tmax)
    
    # Stash initial lr for warmup
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])   
    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    global_step = 0
    # Optional per-step CSV logging
    steps_csv_path = os.path.join("logs", "projector", f"steps_{get_log_name(args)}.csv")
    steps_csv_file = None
    steps_csv_writer = None
    if args.log_every_step:
        steps_csv_file = open(steps_csv_path, mode="w", newline="")
        steps_csv_writer = csv.writer(steps_csv_file)
        steps_csv_writer.writerow(["phase", "epoch", "step", "loss", "avg_loss", "lr", "grad_norm"])  # header

    for epoch in range(1, args.epochs+1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        decoder.train()
        total_tr_loss = 0
        epoch_losses = []
        for i, (images, _) in enumerate(pbar):
            # Linear warmup of LR
            if args.warmup_steps > 0 and global_step < args.warmup_steps:
                warmup_ratio = float(global_step + 1) / float(max(1, args.warmup_steps))
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * warmup_ratio
            with torch.no_grad():
                cls_h = get_clip_image_features(images, clip_model, clip_preprocess)
                tokens_l = get_model_image_features(args.model_name, list(images), model, processor)
            # Compute the loss in float32 for stability
            pred_tokens = decoder(cls_h).float()
            tokens_l = tokens_l.float()
            loss = criterion(pred_tokens, tokens_l)
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss detected at epoch {epoch}, step {i}: {loss.item()}")
                if args.stop_on_nan:
                    raise RuntimeError("Stopping due to NaN/Inf loss during training")
            optimizer.zero_grad()
            loss.backward()
            grad_norm_value = None
            if args.grad_clip_norm and args.grad_clip_norm > 0.0:
                grad_norm = clip_grad_norm_(decoder.parameters(), max_norm=args.grad_clip_norm)
                grad_norm_value = float(grad_norm)
            optimizer.step()

            epoch_losses.append(loss.item())
            total_tr_loss += loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            avg_tr = total_tr_loss / (i + 1)
            pbar.set_postfix({"loss": loss.item(), "total_loss": avg_tr, "lr": current_lr})
            if args.log_every_step:
                logger.info(f"train | epoch={epoch} step={i} loss={loss.item():.6f} avg={avg_tr:.6f} lr={current_lr:.6e} grad_norm={grad_norm_value}")
                if steps_csv_writer is not None:
                    steps_csv_writer.writerow(["train", epoch, i, float(loss.item()), float(avg_tr), float(current_lr), grad_norm_value if grad_norm_value is not None else ""])            
            if args.use_wandb and wandb is not None:
                wandb.log({
                    "phase": "train",
                    "train/loss": float(loss.item()),
                    "train/avg_loss": float(avg_tr),
                    "lr": float(current_lr),
                    "grad_norm": grad_norm_value if grad_norm_value is not None else None,
                    "epoch": epoch,
                    "step": i,
                }, step=global_step)
            global_step += 1
        avg_train_loss = total_tr_loss / len(train_loader)
        logger.info(f"Epoch {epoch} - Train Loss: {sum(epoch_losses) / len(epoch_losses)}")
        train_losses.append(avg_train_loss)
        total_val_loss = 0.0
        decoder.eval()
        pbar = tqdm(val_loader, desc=f"Validating epoch {epoch}")
        for i, (images, _) in enumerate(pbar):   
            with torch.no_grad():
                cls_h = get_clip_image_features(images, clip_model, clip_preprocess)
                tokens_l = get_model_image_features(args.model_name, list(images), model, processor)
                pred_tokens = decoder(cls_h).float()
                # B, T, D = pred_tokens.shape
                # token_idx = torch.randint(0, T, (B, args.no_selected_tokens), device=pred_tokens.device)  # [B, 32]
                # batch_idx = torch.arange(B, device=pred_tokens.device).unsqueeze(1)  # [B, 1]
                # pred_sample = pred_tokens[batch_idx, token_idx]  # [B, 32, D]
                # target_sample = tokens_l[batch_idx, token_idx]   # [B, 32, D]
                # val_loss = criterion(pred_sample, target_sample)
                val_loss = criterion(pred_tokens.float(), tokens_l.float())
                if not torch.isfinite(val_loss):
                    logger.warning(f"Non-finite val loss at epoch {epoch}, step {i}: {val_loss.item()}")
                    if args.stop_on_nan:
                        raise RuntimeError("Stopping due to NaN/Inf loss during validation")
                total_val_loss += val_loss.item()
            
            pbar.set_postfix({"val_loss": val_loss.item(), "total_val_loss": total_val_loss / (i + 1)})
            if args.log_every_step:
                current_lr = optimizer.param_groups[0]["lr"]
                avg_val = total_val_loss / (i + 1)
                logger.info(f"val   | epoch={epoch} step={i} loss={val_loss.item():.6f} avg={avg_val:.6f} lr={current_lr:.6e}")
                if steps_csv_writer is not None:
                    steps_csv_writer.writerow(["val", epoch, i, float(val_loss.item()), float(avg_val), float(current_lr), ""])   
            if args.use_wandb and wandb is not None:
                wandb.log({
                    "phase": "val",
                    "val/loss": float(val_loss.item()),
                    "val/avg_loss": float(total_val_loss / (i + 1)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "epoch": epoch,
                    "step": i,
                }, step=global_step)
        
        logger.info(f"Epoch {epoch} - Validation Loss: {total_val_loss / len(val_loader)}")
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        if args.use_wandb and wandb is not None:
            wandb.log({
                "epoch/train_loss": float(avg_train_loss),
                "epoch/val_loss": float(avg_val_loss),
                "epoch": epoch,
            }, step=global_step)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        # Always save per-epoch checkpoint
        ckpt_path = os.path.join(args.save_projector_dir, get_log_name(args))+f"_epoch_{epoch}.pt"
        torch.save(decoder.state_dict(), ckpt_path)
        print(f"Saved projector model with loss {best_val_loss} to {ckpt_path}")
        
        # Optionally run test after each epoch
        if args.run_test_each_epoch:
            test_cache = args.test_cache_path if args.test_cache_path is not None else args.cache_path
            cmd = [
                sys.executable,
                args.test_script,
                "--model_name", args.model_name,
                "--data_path", args.test_data_path,
                "--cache_path", test_cache,
                "--projector_path", ckpt_path,
                "--target_objects", args.test_target_objects,
            ]
            logger.info(f"Running test script: {' '.join(cmd)}")
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
                test_output = proc.stdout
                logger.info("\n" + test_output)
                # Parse mean projector accuracy from test output
                # Expect line like: "Mean Accuracies for Projector: X"
                mean_match = re.search(r"Mean Accuracies for Projector:\s*([0-9]*\.?[0-9]+)", test_output)
                if mean_match:
                    mean_acc = float(mean_match.group(1))
                    logger.info(f"Parsed Mean Projector Accuracy: {mean_acc}")
                    if args.use_wandb and wandb is not None:
                        wandb.log({"epoch/test_mean_projector_accuracy": mean_acc, "epoch": epoch}, step=global_step)
                else:
                    logger.warning("Could not parse Mean Projector Accuracy from test output.")
            except Exception as e:
                logger.exception(f"Failed running test script: {e}")
            if args.use_wandb and wandb is not None:
                wandb.run.summary["best_val_loss"] = float(best_val_loss)
        # Step scheduler at epoch end (after warmup)
        if scheduler is not None and (args.warmup_steps == 0 or global_step >= args.warmup_steps):
            scheduler.step()
        
    plot_path = os.path.join(args.save_projector_dir, get_log_name(args) + "_loss_curve.png")
    logger.info(f"Train losses: {train_losses}")
    logger.info(f"Val losses: {val_losses}")

    plt.figure(figsize=(18, 6))

    # Train Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train Loss")
    plt.grid(True)

    # Val Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Validation Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved loss curve to {plot_path}")
    logger.info(f"Training completed. Best validation loss: {best_val_loss}")
    if steps_csv_file is not None:
        steps_csv_file.close()


if __name__ == "__main__":
    args = get_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO

    # create folder
    os.makedirs(f"logs", exist_ok=True)
    os.makedirs(f"logs/projector", exist_ok=True)
    os.makedirs(args.save_projector_dir, exist_ok=True)

    logging.basicConfig(format="### %(message)s ###")

    logger = logging.getLogger("HallucinationAttack")
    logger.setLevel(level=logging_level)

    logger.addHandler(logging.FileHandler(f"logs/projector/{get_log_name(args)}.txt", mode='w'))

    # Setting Seed
    set_seed(args.seed)

    logger.info(f"Arguments: {args}")

    # Initialize W&B if requested
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install with `pip install wandb` or run without --use_wandb.")
        run_name = args.wandb_run_name if args.wandb_run_name is not None else get_log_name(args)
        wandb.init(project=args.wandb_project, name=run_name, entity=args.wandb_entity, config=vars(args))

    train_projector(args)

    if args.use_wandb and wandb is not None:
        wandb.finish()

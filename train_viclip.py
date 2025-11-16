import os
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from ViClip.simple_tokenizer import SimpleTokenizer
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as T
import random


# ─── Configuration ─────────────────────────────────────────────────────────
MODEL_DIR             = "/ViClip"
HF_MODEL              = "OpenGVLab/ViCLIP-L-14-hf"
VIDEO_DIR             = "data/raw/to_infer/dedup_ds"
TRAINING_CAPTION_FILE = "data/training_captions.jsonl"
NUM_FRAMES            = 8
TARGET_SIZE           = (224, 224)
DEVICE                = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE            = 112          
EPOCHS                = 100     
EARLY_STOP_PATIENCE   = 15           
LR                    = 1e-4
LORA_R                = 18
LORA_A                = 18
EVAL_STEPS            = 100          # evaluate every N steps
LOG_DIR               = "./runs/finetune_viclip"
NUM_WORKERS           = 12
AUGMENT               = True
AUGMENT_RATE          = 0.8

# ─── Image normalization stats (float32) ──────────────────────────────────
V_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
V_STD  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

# Define spatial augmentations for training
video_augment = T.Compose([
    T.ToPILImage(),           # PILImage → PILImage (identity if already PIL)
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor()              # PIL → FloatTensor
])

def augment_video_frames(frames):
    '''
    1. Apply augmentations return float tensor CHW 0-1
    2. Scale back values to 0-255 and cast to int8
    3. Permute from CHW to HWC (numpy format)
    4. Convert back to numpy array 
    '''
    out = []
    for f in frames:
        if random.random() < AUGMENT_RATE:
            aug_tensor = video_augment(f)
            aug_uint8  = (aug_tensor * 255).to(torch.uint8)
            out.append(aug_uint8.permute(1,2,0).numpy())
        else:
            out.append(f)
    return out


# ─── Video sampling & tensor conversion ────────────────────────────────────

import os, sys
from contextlib import contextmanager

# Supress cv2 errors 
@contextmanager
def suppress_stderr():
    devnull = os.open(os.devnull, os.O_RDWR)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)

def sample_frames(video_path):
    """
    Sample NUM_FRAMES evenly distributed frames from a video file.
    
    Args:
        video_path: Path to the video file
        augment: Whether to apply data augmentation
        
    Returns:
        List of frame arrays in RGB format, or None if video can't be read
    """
    with suppress_stderr():
        video_capture = cv2.VideoCapture(video_path)
        total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if video is valid
        if total_frame_count == 0:
            video_capture.release()
            return None
        
        # Calculate frame indices to sample evenly across the video
        frame_indices = np.linspace(0, total_frame_count - 1, NUM_FRAMES, dtype=int)
        sampled_frames = []
        last_valid_frame = None
        
        for target_frame_idx in frame_indices:
            # Try to read the target frame
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            success, current_frame = video_capture.read()
            
            # If reading fails, try subsequent frames
            search_frame_idx = target_frame_idx
            while not success and search_frame_idx < total_frame_count - 1:
                search_frame_idx += 1
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, search_frame_idx)
                success, current_frame = video_capture.read()
            
            # Handle case where no frame could be read
            if not success:
                if last_valid_frame is None:
                    video_capture.release()
                    return None
                current_frame = last_valid_frame
            else:
                # Convert from BGR to RGB format
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                last_valid_frame = current_frame
            
            sampled_frames.append(current_frame)
        
        video_capture.release()
        
        # Pad with last frame if we don't have enough frames
        while len(sampled_frames) < NUM_FRAMES:
            sampled_frames.append(sampled_frames[-1])

        return sampled_frames

def frames2tensor_train(frames: list[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of video frames to a normalized PyTorch tensor.
    
    Args:
        frames: List of frame arrays in RGB format (H, W, C)
        
    Returns:
        Normalized tensor with shape (C, D, H, W) where:
        - C: channels (3 for RGB)
        - D: number of frames 
        - H, W: height and width (224, 224)
    """
    scaled_rgb_frames = [cv2.resize(f, TARGET_SIZE) for f in frames]
    normalized_frames = np.stack(scaled_rgb_frames, axis=0).astype(np.float32) / 255.0  # (D,H,W,C)
    normalized_frames = (normalized_frames - V_MEAN.astype(np.float32)) / V_STD.astype(np.float32)                            # ImageNet normalization
    return torch.from_numpy(normalized_frames.transpose(3, 0, 1, 2))                   # (C,D,H,W)

# ─── Dataset ───────────────────────────────────────────
class VideoTextDataset(Dataset):
    def __init__(self, lookup_file, video_dir, augment=False):
        self.meta      = [json.loads(l) for l in open(lookup_file)]
        self.video_dir = video_dir
        self.augment   = augment
        self._cache    = {}   # idx → raw frames list

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        item = self.meta[idx]

        # Load & cache raw frames
        if idx not in self._cache:
            filename = item["file_name"]
            path = os.path.join(self.video_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"No file for {filename}")

            frames = sample_frames(path)
            if frames is None:
                raise RuntimeError(f"Could not load frames from {path}")
            self._cache[idx] = frames

        raw_frames = self._cache[idx]

        # Apply per‐fetch augmentation if requested
        frames = augment_video_frames(raw_frames) if self.augment else raw_frames

        # Convert to tensor
        video_tensor = frames2tensor_train(frames)

        return video_tensor, item["captions"]
    
if __name__ == '__main__':
    tensorboard_writer = SummaryWriter(log_dir=LOG_DIR)

    # ─── Load model & tokenizer ─────────────────────────────────────────────
    config     = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True)
    tokenizer  = SimpleTokenizer(bpe_path=config.tokenizer_path)

    # ─── LoRA adapters ───────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type="FEATURE_EXTRACTION", inference_mode=False,
        r=LORA_R, lora_alpha=LORA_A, lora_dropout=0.3, bias="none",
        target_modules=["out_proj","c_fc","c_proj"],
    )
    model = get_peft_model(base_model, lora_cfg).to(DEVICE).train()

    # ─── Datasets & loaders ─────────────────────────────────────────────────
    dataset = VideoTextDataset(TRAINING_CAPTION_FILE, VIDEO_DIR, augment=AUGMENT)
    n_test = int(0.2 * len(dataset))
    n_train = len(dataset) - n_test
    train_ds, eval_ds = random_split(dataset, [n_train, n_test])
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    eval_loader = DataLoader(
        eval_ds,  batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    # ─── Optimizer & scheduler ──────────────────────────────────────────────
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # 5% of total training

    optimizer = torch.optim.AdamW(model.parameters(),  
                                 lr=LR,  
                                 weight_decay=1e-2)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer,  
                                              num_warmup_steps=warmup_steps,  
                                              num_training_steps=total_steps)

    best_loss, no_improve, global_step = float('inf'), 0, 0
    first_eval_metrics = None
    best_eval_metrics = None
    best_loss = float('inf')
    best_step = 0
    should_stop = False

    # ─── Main Training Loop ──────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        for video_batch, text_batch in train_loader:
            video_batch = video_batch.to(DEVICE)
            # ─── Forward Pass ────────────────────────────────────────────────────
            optimizer.zero_grad()
            
            # Prepare video data: [B,C,D,H,W] → [B,D,C,H,W] and enable gradients
            video_tensor = video_batch.permute(0, 2, 1, 3, 4).requires_grad_(True)
            
            # Compute text embeddings on-the-fly using current model weights
            text_embeddings_list = [model.get_text_features(text, tokenizer) for text in text_batch]
            text_features = torch.cat(text_embeddings_list, dim=0).squeeze(1)
            
            # Extract video embeddings through vision encoder
            video_embeddings = model.encode_vision(video_tensor, test=False).float()
            
            # L2 normalize embeddings for cosine similarity computation
            normalized_video_embeddings = video_embeddings / video_embeddings.norm(dim=-1, keepdim=True)
            normalized_text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity matrix: [batch_size, batch_size]
            similarity_logits = normalized_video_embeddings @ normalized_text_embeddings.t()
            
            # Create labels for contrastive learning (diagonal should be 1)
            contrastive_labels = torch.arange(similarity_logits.size(0), device=DEVICE)
            
            # Bidirectional contrastive loss: video→text + text→video
            video_to_text_loss = F.cross_entropy(similarity_logits, contrastive_labels)
            text_to_video_loss = F.cross_entropy(similarity_logits.t(), contrastive_labels)
            total_loss = 0.5 * (video_to_text_loss + text_to_video_loss)
            
            # ─── Backward Pass and Updates ───────────────────────────────────────
            total_loss.backward() # Compute gradients according to loss
            optimizer.step() # Update trainable parameters  
            scheduler.step() # Update lr scheduler  
            global_step += 1 # Update step count
            
            # Log training loss to TensorBoard
            tensorboard_writer.add_scalar("train/loss", total_loss.item(), global_step)

            # ─── Evaluation Phase ────────────────────────────────────────────────────
            if global_step % EVAL_STEPS == 0:
                model.eval()
                evaluation_loss = 0.0
                all_video_embeddings = []
                all_text_embeddings = []
                last_eval_video_batch = None  # To save the last batch for visualization
                
                with torch.no_grad():
                    for eval_video_batch, eval_text_batch in eval_loader:
                        last_eval_video_batch = eval_video_batch  # Save the last batch

                        # Process video embeddings
                        eval_video_tensor = eval_video_batch.to(DEVICE).permute(0, 2, 1, 3, 4)
                        eval_video_embeddings = model.encode_vision(eval_video_tensor, test=True).float()
                        normalized_eval_video_embeddings = eval_video_embeddings / eval_video_embeddings.norm(dim=-1, keepdim=True)
                        all_video_embeddings.append(normalized_eval_video_embeddings.cpu())

                        # Compute text embeddings on-the-fly using current model weights (per-sample)
                        eval_text_embeddings_list = [model.get_text_features(text, tokenizer) for text in eval_text_batch]
                        eval_text_embeddings = torch.cat(eval_text_embeddings_list, dim=0).squeeze(1)
                        normalized_eval_text_embeddings = eval_text_embeddings / eval_text_embeddings.norm(dim=-1, keepdim=True)
                        all_text_embeddings.append(normalized_eval_text_embeddings)

                        # Compute batch-wise evaluation loss
                        batch_similarity_matrix = normalized_eval_video_embeddings @ normalized_eval_text_embeddings.t()
                        batch_labels = torch.arange(batch_similarity_matrix.size(0), device=batch_similarity_matrix.device)
                        batch_video_to_text_loss = F.cross_entropy(batch_similarity_matrix, batch_labels)
                        batch_text_to_video_loss = F.cross_entropy(batch_similarity_matrix.t(), batch_labels)
                        evaluation_loss += 0.5 * (batch_video_to_text_loss + batch_text_to_video_loss).item()
        
                # ─── Compute Retrieval Metrics ───────────────────────────────────
                evaluation_loss /= len(eval_loader)
                concatenated_video_embeddings = torch.cat(all_video_embeddings, dim=0).to(DEVICE)
                concatenated_text_embeddings = torch.cat(all_text_embeddings, dim=0).to(DEVICE)
                full_similarity_matrix = concatenated_video_embeddings @ concatenated_text_embeddings.t()
                retrieval_ranks = []
                for video_idx, similarity_scores in enumerate(full_similarity_matrix):
                    sorted_indices = torch.argsort(similarity_scores, descending=True)
                    correct_text_rank = (sorted_indices == video_idx).nonzero().item() + 1
                    retrieval_ranks.append(correct_text_rank)
                ranks_tensor = torch.tensor(retrieval_ranks, dtype=torch.float32)
                recall_at_1 = (ranks_tensor <= 1).float().mean().item()
                recall_at_5 = (ranks_tensor <= 5).float().mean().item()
                recall_at_10 = (ranks_tensor <= 10).float().mean().item()
                median_rank = ranks_tensor.median().item()

                print(f"Epoch {epoch} Step {global_step} Eval loss: {evaluation_loss:.4f} | "
                      f"R@1={recall_at_1:.3f} R@5={recall_at_5:.3f} R@10={recall_at_10:.3f}")
                tensorboard_writer.add_scalar("eval/recall10", recall_at_10, global_step)
                tensorboard_writer.add_scalar("eval/loss", evaluation_loss, global_step)
                tensorboard_writer.add_scalar("eval/recall1", recall_at_1, global_step)
                tensorboard_writer.add_scalar("eval/recall5", recall_at_5, global_step)
                tensorboard_writer.add_scalar("eval/med_rank", median_rank, global_step)

                if global_step // EVAL_STEPS == 1:  # First evaluation
                    first_eval_metrics = {
                        "loss": evaluation_loss,
                        "recall_at_1": recall_at_1,
                        "recall_at_5": recall_at_5,
                        "recall_at_10": recall_at_10,
                        "median_rank": median_rank,
                        "step": global_step
                    }

                if best_loss > evaluation_loss:
                    best_loss = evaluation_loss
                    no_improve = 0
                    best_step = global_step
                    best_eval_metrics = {
                        "loss": evaluation_loss,
                        "recall_at_1": recall_at_1,
                        "recall_at_5": recall_at_5,
                        "recall_at_10": recall_at_10,
                        "median_rank": median_rank,
                        "step": global_step
                    }
                    model.save_pretrained('./best_lora_model')
                    print(f"Best model saved at step {global_step} with eval loss {evaluation_loss:.4f}")
                else:
                    no_improve += 1
                    if no_improve >= EARLY_STOP_PATIENCE:
                        print(f"Early stopping triggered at step {global_step}")
                        should_stop = True
            model.train()
            if should_stop:
                        break
        if should_stop:
            break

    model.save_pretrained('./final_lora_model')
    tensorboard_writer.close()

    # ─── Training Summary ─────────────────────────────────────────────────
    print("\n--- Training Summary ---")
    print(f"LoRa R: {LORA_R} LoRa A: {LORA_A}")
    print(f"Augmentation: {AUGMENT} Augmentation Rate: {AUGMENT_RATE}")
    if first_eval_metrics:
        print(f"First Eval @ step {first_eval_metrics['step']}: "
              f"Loss={first_eval_metrics['loss']:.4f}, "
              f"R@1={first_eval_metrics['recall_at_1']:.3f}, "
              f"R@5={first_eval_metrics['recall_at_5']:.3f}, "
              f"R@10={first_eval_metrics['recall_at_10']:.3f}, "
              f"MedR={first_eval_metrics['median_rank']:.1f}")
    if best_eval_metrics:
        print(f"Best Eval @ step {best_eval_metrics['step']}: "
              f"Loss={best_eval_metrics['loss']:.4f}, "
              f"R@1={best_eval_metrics['recall_at_1']:.3f}, "
              f"R@5={best_eval_metrics['recall_at_5']:.3f}, "
              f"R@10={best_eval_metrics['recall_at_10']:.3f}, "
              f"MedR={best_eval_metrics['median_rank']:.1f}")

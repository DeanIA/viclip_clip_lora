import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from peft import get_peft_model, LoraConfig
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers.models.auto.modeling_auto import AutoModel
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTokenizer
import json 
import os
import glob 
import random
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
import numpy as np

# Config 
MODEL = "openai/clip-vit-large-patch14-336"
PROCESSOR = "openai/clip-vit-large-patch14-336"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LORA_R = 128
LORA_A = 128
TARGET_SIZE = (224, 224)
LOG_DIR = "./runs/finetune_clip"
LOOKUP_FILE = "data/training_captions_images.jsonl"
VIDEO_DIR = "assets/images"
BATCH_SIZE = 128
NUM_WORKERS = 32
LR = 1e-4
EPOCHS = 500
EVAL_STEPS = 100
EARLY_STOP_PATIENCE = 10
AUGMENT_RATE = 0.5

# Load model and processors from huggingface 
base_model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float16)
img_processor = CLIPImageProcessor.from_pretrained(PROCESSOR)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

# Attach LoRa adapters 
lora_cfg = LoraConfig(
  task_type="FEATURE_EXTRACTION",
  inference_mode=False,
  r=LORA_R, lora_alpha=LORA_A, lora_dropout=0.1,
  bias="none",
  target_modules=["visual_projection", "text_projection"],
)

model = get_peft_model(base_model, lora_cfg).to(DEVICE).train()

# Define image augmentations 
img_augment = T.Compose([
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor()              # PIL → FloatTensor
])

def augment_images(img):
    """
    Apply augmentations to an image with probability AUGMENT_RATE.
    Returns a numpy array (HWC, uint8) if augmented, else the original image as numpy array.
    """
    if random.random() < AUGMENT_RATE:
        aug_tensor = img_augment(img)
        aug_uint8  = (aug_tensor * 255).to(torch.uint8)
        aug_uint8 = aug_uint8.permute(1,2,0).numpy()
        augmented_image = Image.fromarray(aug_uint8)
        return augmented_image
    else:
        original_image = np.array(img)
        original_image = Image.fromarray(original_image)
        return original_image

# Define dataset that loads images and caption pairs
class ImageTextDataset(Dataset):
  def __init__(self, 
               lookup_file: str, 
               image_dir: str, 
               augment: bool = True):
      """
      Loads JSONL metadata, and samples/augments frames *on-the-fly*
      each time __getitem__ is called.  That way every epoch sees fresh
      temporal + spatial jitter.

      Args:
          lookup_file: Path to JSONL file containing video metadata
          image_dir: Directory containing image files
          augmentor: Optional function to apply augmentations
      """
      self.meta = [json.loads(l) for l in open(lookup_file, "r")]
      self.image_dir = image_dir
      self.augment = augment 

  def __len__(self) -> int:
    return len(self.meta)
  
  def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
    """
    Get a image-text pair at the given index.
    
    Returns:
        Tuple of (image_tensor, text_description)
    """
    json_item = self.meta[idx]

    # find the image file
    stem = json_item["file_name"]
    image_path = os.path.join(self.image_dir, stem)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No file found for {stem}")
    image = Image.open(image_path).convert("RGB")

    # choose either the augmented image or the original
    if self.augment:
        image_to_process = augment_images(image)
    else:
        image_to_process = image

    batch = img_processor(images=image_to_process, return_tensors="pt") # type: ignore
    pixel_values = batch["pixel_values"]

    if pixel_values is None:
        raise RuntimeError(f"Could not load frames from {image_path}")
    
    text_description = json_item["captions"]
    return pixel_values, text_description

# Dataset & loaders
dataset = ImageTextDataset(LOOKUP_FILE, VIDEO_DIR, augment=True)
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

# Optimizer & scheduler 
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(0.05 * total_steps)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=LR,
                              weight_decay=1e-2)

scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)

# Init tensorboard writer 
tensorboard_writer = SummaryWriter(log_dir=LOG_DIR)

# Training loop counters 
best_loss, no_improve, global_step = float('inf'), 0, 0
first_eval_metrics = None
best_eval_metrics = None
best_loss = float('inf')
best_step = 0

for epoch in range(EPOCHS):
   for image_batch, text_batch in train_loader:
      image_batch = image_batch.to(DEVICE)
      
      # Forward pass 
      optimizer.zero_grad() # restart grad 

      # Tokenize text batch
      inputs = tokenizer(
          text_batch,
          padding=True,
          truncation=True,
          return_tensors="pt"
      ).to(DEVICE)

      # Compute text embeddings using CLIP model
      text_embeds = model.get_text_features(**inputs)

      # Extract image embeddings 
      image_embeddings = model.encode_image(image_batch)

      # L2 normalize embeddings for similarity computaiton 
      normalized_image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
      normalized_text_embeddings = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
      # Compute similarity 
      similarity_logits = normalized_image_embeddings @ normalized_text_embeddings.t() # .t() changes the order of the dimension to allow matrix multiplication

      # Create labels for contrastive learning
      contrastive_labels = torch.arange(similarity_logits.size(0), device=DEVICE)

      # Bidirectional contrastive loss
      image_to_text_loss = F.cross_entropy(similarity_logits, contrastive_labels) 
      text_to_image_loss = F.cross_entropy(similarity_logits.t(), contrastive_labels)
      total_loss = 0.5 * (image_to_text_loss + text_to_image_loss)

      # Backward pass and updates 
      total_loss.backward() # Compute gradients according to loss
      optimizer.step() # Update trainable parameters
      scheduler.step() # Update lr scheduler 
      global_step += 1 # Update step count 

      # Log training loss to tensorboard 
      tensorboard_writer.add_scalar("train/loss", total_loss.item(), global_step)

      # Eval phase 
      if global_step % EVAL_STEPS == 0:
        model.eval()
        evaluation_loss = 0.0
        all_image_embeddings = []
        all_text_embeddings = []
        last_eval_image_batch = None # To save the last batch for visualization 

        with torch.no_grad(): # Eval doesn't update gradients like train 
            for eval_image_batch, eval_text_batch in eval_loader:
                last_eval_video_batch = eval_image_batch  # Save the last batch

                # Process image embeddings 
                eval_image_tensor = eval_image_batch.to(DEVICE)
                eval_image_embeddings = model.get_image_features(eval_image_tensor).float()
                normalized_eval_image_embeddings = eval_image_embeddings / eval_image_embeddings.norm(dim=-1, keepdim=True)
                all_image_embeddings.append(normalized_eval_image_embeddings.cpu())

                # new: batch-tokenize & encode
                eval_inputs = tokenizer(
                    eval_text_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(DEVICE)
                eval_text_embeddings = model.get_text_features(**eval_inputs)
                normalized_eval_text_embeddings = (
                    eval_text_embeddings
                    / eval_text_embeddings.norm(dim=-1, keepdim=True)
                )
                all_text_embeddings.append(normalized_eval_text_embeddings.cpu())

                # Compute batch-wise evaluation loss
                batch_similarity_matrix = normalized_eval_image_embeddings @ normalized_eval_text_embeddings.t()
                batch_labels = torch.arange(batch_similarity_matrix.size(0), device=batch_similarity_matrix.device)
                batch_image_to_text_loss = F.cross_entropy(batch_similarity_matrix, batch_labels)
                batch_text_to_image_loss = F.cross_entropy(batch_similarity_matrix.t(), batch_labels)
                evaluation_loss += 0.5 * (batch_image_to_text_loss + batch_text_to_image_loss).item()

                evaluation_loss /= len(eval_loader)
                concatenated_image_embeddings = torch.cat(all_image_embeddings, dim=0).to(DEVICE)
                concatenated_text_embeddings = torch.cat(all_text_embeddings, dim=0).to(DEVICE)
                full_similarity_matrix = concatenated_image_embeddings @ concatenated_text_embeddings.t()
                retrieval_ranks = []
                for image_idx, similarity_scores in enumerate(full_similarity_matrix):
                    sorted_indices = torch.argsort(similarity_scores, descending=True)
                    correct_text_rank = (sorted_indices == image_idx).nonzero().item() + 1
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
                        tensorboard_writer.close()
                        break  # or exit(0)
                model.train()

model.save_pretrained('./final_lora_model')
tensorboard_writer.close()

# ─── Training Summary ─────────────────────────────────────────────────
print("\n--- Training Summary ---")
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


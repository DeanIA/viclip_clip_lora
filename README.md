# CLIP & ViCLIP for Visual Search

CLIP and ViCLIP enable semantic search across visual content and support tasks like deduplication. By encoding images and text into a shared embedding space, these models let you query large video archives by semantic meaning rather than metadata alone.

## The Domain Specificity Problem

Both models are trained on generic datasets that often lack the specificity needed for specialized domains. Off-the-shelf embeddings may miss nuances critical to your use case, leading to suboptimal search results and deduplication performance.

## Improving Embeddings with LoRA Fine-tuning

We can improve model performance by fine-tuning on domain-specific data. Rather than retraining the entire model (computationally expensive), LoRA fine-tuning adds lightweight, trainable adapter layers on top of frozen base weights. This approach requires minimal additional parameters while substantially improving embedding quality for your specific domain.

![Diagram from Hugging Face](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png)

### What is LoRA Fine-tuning?

LoRA (Low-Rank Adaptation) works by injecting small, trainable matrices into the model's attention layers. Instead of updating millions of parameters, you train only these adapters, which are then combined with the original weights at inference time. This makes fine-tuning fast, memory-efficient, and practical for domain adaptation without the cost of full model retraining.

## Models Used

- `OpenGVLab/ViCLIP-B-16-hf`
- `openai/clip-vit-large-patch14-336`

## Fine-tuning Process

### Dataset Creation

Generate custom clip-caption pairs using a large language model like Gemini. Ensure clip filenames contain an index indicating their original position within source videos for proper alignment.

### Frame Sampling

The `VideoTextDataset` class handles the data pipeline. For each 30-second clip, it identifies the correct segment using the filename index, calculates evenly distributed frame indices, and reads frames in RGB format. If frames are missing, the last successfully read frame is duplicated to maintain consistent frame counts per clip. Output is a list of tuples containing eight frame tensors and their corresponding text captions.

### Training Configuration

Key parameters for optimal performance:

- LoRA Rank: 32
- LoRA Alpha: 64
- Batch Size: 64
- Learning Rate: 1e-4
- Evaluation Frequency: Every 100 steps
- Early Stopping: Stops if evaluation loss doesn't improve for 10 consecutive epochs
- LR Scheduler: 5% warm-up followed by cosine decay

### Loss and Monitoring

Training uses contrastive loss to align video and text embeddings in a shared space. L2-normalized embeddings are compared using cosine similarity, with bidirectional alignment ensuring both video-to-text and text-to-video matching. Monitor performance using TensorBoard to track contrastive loss, top-1/5/10 retrieval accuracy, and visual confirmation of training progress.
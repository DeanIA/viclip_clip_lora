# CLIP & ViCLIP

CLIP and ViCLIP enable semantic search across visual content and support tasks like deduplication. By encoding images and text into a shared embedding space, these models let you query large video archives by semantic meaning rather than metadata alone.

## The Domain Specificity Problem

Both models are trained on generic datasets that often lack the specificity needed for specialized domains. Off-the-shelf embeddings may miss nuances critical to your use case, leading to suboptimal search results and deduplication performance.

## Improving Embeddings with LoRA Fine-tuning

We can improve model performance by fine-tuning on domain-specific data. Rather than retraining the entire model (computationally expensive), LoRA fine-tuning adds lightweight, trainable adapter layers on top of frozen base weights. This approach requires minimal additional parameters while substantially improving embedding quality for your specific domain.

### What is LoRA Fine-tuning?

LoRA (Low-Rank Adaptation) works by injecting small, trainable matrices into the model's attention layers. Instead of updating millions of parameters, you train only these adapters, which are then combined with the original weights at inference time. This makes fine-tuning fast, memory-efficient, and practical for domain adaptation without the cost of full model retraining.

## Models Used

- `OpenGVLab/ViCLIP-B-16-hf`
- `openai/clip-vit-large-patch14-336`
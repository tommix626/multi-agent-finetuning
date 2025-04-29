#!/usr/bin/env python3
"""
./scripts/mixture_inference.py

Inference on your trained X-LoRA mixer (full split):
  • Accuracy
  • Avg cross-entropy loss
  • Perplexity
"""
import os
import sys
import torch
import xlora
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────────────
# make data/mmlu visible as a module
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "data", "mmlu"))
from mmludataset import pre_process_data
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Config
    CHECKPOINT_DIR = "./checkpoints/low-mi-temp-epoch-9-mixed"
    BASE_MODEL_ID  = "EleutherAI/gpt-neo-125M"
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE     = 4  # Reduced from 8
    
    # Check CUDA availability first and print info
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Clear cache before loading model
        torch.cuda.empty_cache()
    else:
        print("CUDA not available, using CPU")
    
    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"       # CHANGED: ensure left-padding for decoder-only model

    try:
        # 3) Load base model directly to device if CUDA available
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,  # Use auto device mapping
        )
        # disable past-key-value caching as X-LoRA requires
        base_model.config.use_cache = False
        
        # If device_map is None (CPU), explicitly move to device
        if not torch.cuda.is_available():
            base_model = base_model.to(DEVICE)
            
        print(f"Model loaded successfully on {DEVICE}")

        # 4) Collect your expert adapter dirs
        adapter_paths = {
            d: os.path.join(CHECKPOINT_DIR, d)
            for d in os.listdir(CHECKPOINT_DIR)
            if d.startswith("adapter_for_expert_")
        }
        print(f"Found {len(adapter_paths)} adapters")

        # 5) Rehydrate your X-LoRA model (mixer + adapters)
        xlo_model = xlora.from_pretrained(
            CHECKPOINT_DIR,           # contains xlora_classifier.safetensors, xlora_config.json, etc.
            model=base_model,
            adapters=adapter_paths,   # explicit dict so adapters ≠ None
            device=DEVICE,
            verbose=True,             # Enable verbose mode to see loading progress
        )
        print("X-LoRA model loaded successfully")

        # 6) Prepare the test DataLoader ("full" = expert + mixer)
        _, _, test_loader = pre_process_data(
            model_name = BASE_MODEL_ID,
            batch_size = BATCH_SIZE,
            device     = DEVICE,
            peft_config= None,
            mode       = "full",
        )
        print(f"Test data loaded with batch size {BATCH_SIZE}")

        # 7) Eval loop
        xlo_model.eval()
        total_loss, num_batches = 0.0, 0
        correct, total = 0, 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}...")
                    
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["labels"].to(DEVICE)

                # forward → cross-entropy loss
                outputs = xlo_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
                num_batches += 1

                # greedy generate up to 50 tokens
                gen_ids = xlo_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,                               # CHANGED: pass attention mask
                    max_new_tokens=50,
                    max_length=input_ids.shape[1] + 50,                          # CHANGED: override default max_length
                    pad_token_id=tokenizer.eos_token_id
                )
                prompt_len = input_ids.size(1)

                for i in range(gen_ids.size(0)):
                    # get generated continuation
                    gen_tokens = gen_ids[i][prompt_len:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    pred_tok = gen_text.split()[0] if gen_text else ""

                    # get true answer tokens
                    ref_ids  = labels[i][labels[i] != -100]
                    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True).strip()
                    ref_tok = ref_text.split()[0] if ref_text else ""

                    if pred_tok and pred_tok == ref_tok:
                        correct += 1
                    total += 1

        # 8) Compute metrics
        avg_loss   = total_loss / num_batches if num_batches else float("nan")
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if num_batches else float("nan")
        accuracy   = (correct / total * 100) if total else float("nan")

        # 9) Print
        print(f"Accuracy:          {accuracy:.2f}%")
        print(f"Avg Cross-Entropy: {avg_loss:.4f}")
        print(f"Perplexity:        {perplexity:.2f}")
        
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        print("\nGPU Memory information:")
        if torch.cuda.is_available():
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print("\nTry one of these solutions:")
        print("1. Use 'export CUDA_VISIBLE_DEVICES=X' to select a specific GPU")
        print("2. Try 'export CUDA_LAUNCH_BLOCKING=1' for better error messages")
        print("3. Check for other processes using 'nvidia-smi'")
        print("4. Reduce batch size further")
        print("5. Restart your Python environment")

if __name__ == "__main__":
    main()

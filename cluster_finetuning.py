"""main entry point of training script"""
import torch
from peft import LoraConfig, TaskType
from training.callbacks import ModelCheckpoint, EarlyStopping
import os
from data.mmlu.mmludataset import pre_process
from training.cluster_perplexity_trainer import TrainerConfig,ExpertTrainer



def main():
    # 1. Hyperparameters and paths
    model_name = "EleutherAI/pythia-70m"
    batch_size = 8
    num_epochs = 3
    num_experts = 4
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./checkpoints"

    # 2. Define LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )

    # 3. Prepare dataset and base model with PEFT
    print("[Main] Loading data and model...")
    peft_model, train_loader, _, _ = pre_process(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        peft_config=peft_config
    )

    # 4. Callbacks
    callbacks = [
        ModelCheckpoint(save_dir=save_dir, monitor="val_loss", mode="min"),
        EarlyStopping(patience=2, monitor="val_loss", mode="min")
    ]

    # 5. Trainer configuration
    config = TrainerConfig(
        model_name=model_name,
        num_experts=num_experts,
        lr=learning_rate,
        epochs=num_epochs,
        device=device,
        callbacks=callbacks
    )

    # 6. Instantiate trainer
    trainer = ExpertTrainer(
        training_config=config,
        dataloader=train_loader,
        peft_config=peft_config
    )

    # 7. Train
    trainer.train()


if __name__ == "__main__":
    main()

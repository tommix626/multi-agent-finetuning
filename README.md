# Multi-Agent Finetuning

This repository provides a framework for fine-tuning large language models (LLMs) in multi-agent environments. It includes tools for training, evaluation, and deployment across distributed systems.

## Repository Structure

- **configs/**: Configuration files for training and evaluation setups.
- **data/mmlu/**: Datasets related to the MMLU benchmark.
- **evaluation/**: Scripts and tools for assessing model performance.
- **models/**: Pretrained models and model architecture definitions.
- **scripts/**: Utility scripts for data preprocessing and other tasks.
- **slurm_jobs/**: SLURM job scripts for managing distributed training.
- **training/**: Core training routines and modules.
- **utils/**: Helper functions and utilities used across the project.
- **cluster_finetuning.py**: Main script for initiating cluster-based fine-tuning.
- **cluster_training_with_config.py**: Script for training using configuration files.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU(s)
- SLURM workload manager (for distributed training)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tommix626/multi-agent-finetuning.git
   cd multi-agent-finetuning
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To initiate training using a configuration file:

```bash
python cluster_training_with_config.py --config configs/your_config.yaml
```

### Evaluation

After training, evaluate the model's performance:

```bash
python evaluation/evaluate_model.py --config configs/your_config.yaml
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

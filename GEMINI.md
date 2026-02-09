# Project Context: CLIP-CAER (Context-Aware Academic Emotion Recognition)

## Project Overview
This project implements **CLIP-CAER**, a context-aware academic emotion recognition method based on **CLIP**. It is designed to recognize student learning states (e.g., focused, distracted) by leveraging contextual information.

**Key Features:**
*   **Dual-Stream Architecture:** Processes both Face (expressions) and Context (body/environment) streams.
*   **Expression-Aware Adapter (EAA):** Fine-tunes CLIP for subtle facial details.
*   **Instance-Enhanced Classifier (IEC):** Blends instance features with text prototypes.
*   **Advanced Losses:** Mutual Information (MI) Loss, Decorrelation (DC) Loss, Semantic LDL Loss.
*   **Temporal Modeling:** Attention Pooling for temporal frame aggregation.

## Key Files & Structure

*   **`train.sh`**: The primary entry point for training. Sets environment variables and invokes `main.py` with configuration arguments.
*   **`valid.sh`**: Entry point for evaluation.
*   **`main.py`**: Handles argument parsing, environment setup, model building, and initiates the training/evaluation loop.
*   **`trainer.py`**: Contains the core `Trainer` class, managing the training epoch logic, loss calculation (including custom MI/DC losses), and metric logging (WAR, UAR).
*   **`dataloader/`**: Contains `video_dataloader.py` for loading the RAER dataset.
*   **`models/`**:
    *   `Generate_Model.py`: Main model definition.
    *   `Adapter.py`: Implementation of the Expression-Aware Adapter.
    *   `Text.py`: Text encoding and prompt learning modules.
*   **`utils/loss.py`**: Custom loss functions (SemanticLDLLoss, MILoss, DCLoss).

## Usage

### Environment
*   **Python:** 3.8
*   **PyTorch:** 2.2.2
*   **CUDA:** 12.4 (compatible with 12.1 wheels)

### Running the Project

**1. Training:**
Execute the training script. You can modify arguments within this file.
```bash
bash train.sh
```
*   **Under the hood:** Runs `python main.py --mode train ...`
*   **Key Arguments:**
    *   `--exper-name`: Name for the experiment (creates output folder).
    *   `--lambda_mi`, `--lambda_dc`: Weights for MI and DC losses.
    *   `--slerp-weight`: IEC interpolation factor.
    *   `--lr-adapter`: Learning rate for the adapter.

**2. Evaluation:**
```bash
bash valid.sh
```

## Development Conventions

*   **Configuration:** The project relies heavily on command-line arguments (parsed in `main.py`) for all hyperparameters and feature toggles.
*   **Logging:** Outputs are saved to `outputs/<experiment_name>/`.
    *   `log.txt`: Text logs of training progress.
    *   `log.png`: Training curves.
    *   `confusion_matrix.png`: Visual confusion matrix.
*   **Debugging:** The `Trainer` class saves sample prediction images to `debug_predictions/` for visual inspection of model performance.
*   **Code Style:** Standard PyTorch practices.
*   **Data:** Expects `RAER` dataset structure with specific annotation files (paths defined in `train.sh`).

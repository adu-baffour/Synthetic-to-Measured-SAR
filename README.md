# Towards Fully Synthetic Training: Exploring Data Augmentations for Synthetic-to-Measured SAR in Automatic Target Recognition

![Project Structure](assets/project_structure.png)

## Abstract

The limited availability of measured Synthetic Aperture Radar (SAR) images challenges robust Automatic Target Recognition (ATR) system development due to high costs and time requirements. Synthetic SAR images offer an alternative, but discrepancies between synthetic and measured data impede real-world generalization. 

This project explores data augmentation techniques to train effective SAR-ATR models using only synthetic data, achieving high performance on measured SAR imagery. We introduce image augmentation methods that simulate key radar signal components:  
- **Target Variation**: Color jitter  
- **System Characteristics**: Gaussian noise  
- **Environmental Effects**: Random erasing  

Our experiments showed that combining Gaussian noise and random erasing with the ConvNeXt-base model achieved a promising **92.01% accuracy** on measured SAR data while training exclusively on synthetic data. Qualitative analyses further demonstrated improved class separation and feature learning, emphasizing the potential of augmentation strategies to bridge the synthetic-to-measured domain gap.

The repository includes model checkpoints and code for reproducing results.  

---

## Features
- Data augmentation strategies for SAR domain adaptation
- Benchmark results with multiple deep learning architectures
- Tools for qualitative analysis: confusion matrices, t-SNE visualizations, and class activation maps
- Fully reproducible training pipeline

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required Python packages: Install via `pip install -r requirements.txt`

### Repository Structure
```plaintext
SYNTHETIC-TO-MEASURED_ATR/
├── dataset/                 # Dataset preprocessing scripts
├── results/                 # Folder for saving experimental results
├── __init__.py              # Package initialization
├── dataset.py               # Dataset loaders and transformations
├── main.py                  # Main script to run experiments
├── model.py                 # Deep learning model definitions
├── playground.ipynb         # Notebook for experimentation and visualization
├── trainer.py               # Model training and evaluation scripts
├── util.py                  # Utility functions
```
---
### Visualizations
Use the `playground.ipynb` notebook to generate qualitative analyses:
- Augmentated images
- Class activation maps
---
### Citation
If you find this repository helpful, please cite our paper:
```plaintext
@article{adu2025synthetic,
  title={Towards Fully Synthetic Training: Exploring Data Augmentations for Synthetic-to-Measured SAR in Automatic Target Recognition},
  author={Adu-Baffour, Isaac Osei Agyemang, Isaac Adjei-Mensah, Raphael Elimeli Nuhoho},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2025}
}
```

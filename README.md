# Towards Fully Synthetic Training: Exploring Data Augmentations for Synthetic-to-Measured SAR in Automatic Target Recognition

## Abstract

The limited availability of measured Synthetic Aperture Radar (SAR) images challenges robust Automatic Target Recognition (ATR) system development due to high costs and time requirements. Synthetic SAR images offer an alternative, but discrepancies between synthetic and measured data impede real-world generalization. This paper presents data augmentation techniques to train effective SAR-ATR models using only synthetic data, achieving high performance on measured SAR imagery. We introduce image augmentation methods that simulate key radar signal components: target variation via color jitter, system characteristics through Gaussian noise, and environmental effects with random erasing. We evaluated these techniques individually and in combination across various deep learning architectures. Our experiments showed that applying Gaussian noise and random erasing with the ConvNeXt-base model achieved a promising accuracy of 92.01% on measured SAR data while training exclusively on synthetic data. Qualitative analyses using confusion matrices, t-SNE visualizations, and class activation maps further show improved class separation and feature learning, especially for challenging targets. These findings highlight the potential of effective augmentation strategies to bridge the synthetic-to-measured domain gap, offering a scalable solution for SAR-ATR systems where measured data are limited or unavailable. The model checkpoints and code are available at https://github.com/adu-baffour/Synthetic-to-Measured-SAR/.

---

 ![alt text](https://github.com/adu-baffour/Synthetic-to-Measured-SAR/blob/main/imgs/architecture.png?raw=true)


## Getting Started
The repository includes code for reproducing results.  

### Prerequisites
- Python 3.8+
- Required Python packages: Install via `pip install -r requirements.txt`

### Repository Structure
```plaintext
SYNTHETIC-TO-MEASURED_ATR/
├── dataset/                 # Dataset image files
├── results/                 # Folder for saving experimental results
├── __init__.py              # Package initialization
├── dataset.py               # Dataset loaders and transformations
├── main.py                  # Main script to run experiments
├── model.py                 # Deep learning model definitions
├── playground.ipynb         # Notebook for qualitative visualization
├── trainer.py               # Model training and evaluation scripts
├── util.py                  # Utility functions
```
---
### Visualizations
Use the `playground.ipynb` notebook to generate qualitative analyses:
##### Augmentated images
  ![alt text](https://github.com/adu-baffour/Synthetic-to-Measured-SAR/blob/main/imgs/augmentation.png?raw=true)
---
##### t-nse
![alt text](https://github.com/adu-baffour/Synthetic-to-Measured-SAR/blob/main/imgs/tsne.png?raw=true)
---
##### Class activation maps
![alt text](https://github.com/adu-baffour/Synthetic-to-Measured-SAR/blob/main/imgs/heatmap.png?raw=true)
---
### Citation
If you find this repository helpful, please cite our paper:
```plaintext
@INPROCEEDINGS{11105313,
  author={Baffour, Adu Asare and Osei Agyemang, Isaac and Adjei-Mensah, Isaac and Nuhoho, Raphael Elimeli},
  booktitle={2025 IEEE World AI IoT Congress (AIIoT)}, 
  title={Towards Fully Synthetic Training: Exploring Data Augmentations for Synthetic-to-Measured SAR in Automatic Target Recognition}, 
  year={2025},
  volume={},
  number={},
  pages={0008-0017},
  keywords={Training;Accuracy;Target recognition;Gaussian noise;Radar imaging;Data augmentation;Radar polarimetry;Data models;Time measurement;Synthetic aperture radar;Synthetic Aperture Radar;Automatic Target Recognition;Data Augmentation;Deep Learning},
  doi={10.1109/AIIoT65859.2025.11105313}}
```

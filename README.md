<h1 align="center">PIPO_net</h1>

## 📰 News
- [02/01/2026] The code for reproducing our multimodal myocardial pathology segmentation results on the MyoPS++ and MyoPS++ datasets is now available. This release is intended for peer review and reproducibility verification. 

## 📑 Overview
Myocardial pathology segmentation is an important task in medical image analysis. Existing methods typically rely on multi-modal magnetic resonance imaging (MRI) to achieve accurate segmentation. However, modality absence is common in real-world
acquisitions, leading to a substantial decline in segmentation performance. To address this challenge, we propose Prompt-Guided Modality Completion (PMC). PMC introduces input prompts and feature prompts at the image and feature levels, respectively, enabling multi-granularity modality completion and effectively handling diverse
missing-modality scenarios. The method can be seamlessly integrated into different baseline networks. Extensive experiments on four baseline models and two myocardial pathology segmentation datasets demonstrate that PMC consistently enhances segmentation accuracy and robustness. For example, on the MyoPS++ dataset, the integration
of PMC across four different baseline networks consistently improves Scar and Edema segmentation accuracy, with gains ranging from 1.21% to 2.16%. Codes are available at https://github.com/zzzzzzl24/PIPO_net.

## ⚙️ Quick Start

### 1. Preparation
First, clone this repository to your local machine and install the required dependencies (`torch`, `torchvision`, `numpy`, and `open_clip`):

```bash
git clone git@github.com:zzzzzzl24/PIPO_net.git 
cd PIPO_net
pip install -r requirements.txt
```
### 2. Model Training & Inference

The project uses the following directory structure:
```bash
PIPO_net/
├── MyoPS++_dataset/     # place MyoPS++ dataset here
│   ├── Raw_data/         # raw bssfp, t2w, lge, label                    
│   └── Processed_data/   # preprocessed data for training/testing
├── preprocess_data.py    # preprocessing script
├── train.py              # training script
├── test.py               # inference script
├── requirements.txt      # Reproducible environment definition
└── README.md
```

Now, try the model with just a few lines of code:

- Run the train script on MyoPS++ dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES= "0" python train.py --dataset QS_Seg_MyoPS --vit_name R50
```

- Run the test script on MyoPS++ dataset. The testing results reported in the manuscript can be reproduced.

```bash
python test.py --dataset QS_Seg_MyoPS --vit_name R50
```

<!-- ## Citations

```bibtex
@article{

}
``` -->

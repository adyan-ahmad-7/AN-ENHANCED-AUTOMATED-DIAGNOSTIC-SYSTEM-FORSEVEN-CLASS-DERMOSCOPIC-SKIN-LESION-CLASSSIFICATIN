# Skin Lesion Classification Using Transfer Learning

A comparative deep-learning project for **seven-class skin-lesion classification** on the HAM10000 dataset. The repository contains three model pipelines that follow the same overall workflow and differ primarily in the CNN backbone:

- **ResNet50**
- **VGG16**
- **InceptionV3**

> **Important:** The detailed pipeline inspected for this README was the InceptionV3 notebook. The ResNet50 and VGG16 notebook names and their exact layer configurations should be checked against the corresponding notebooks before publication. Do not assume that every model-specific setting is identical unless the code confirms it.

## Project overview

The project uses transfer learning to classify dermatoscopic images into the seven HAM10000 diagnostic categories. The workflow includes metadata loading, image-path mapping, class balancing, stratified train/validation/test splitting, image preprocessing and augmentation, model training, held-out test evaluation, visualisation, and model/artifact export.

### Dataset

- **Dataset:** Skin Cancer MNIST: HAM10000
- **Source:** [Kaggle — Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Original dataset size:** approximately 10,015 images
- **Task:** Seven-class image classification
- **Labels used by the notebook:**

| Label | Diagnostic category |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma / Bowen's disease |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

The notebook expects the Kaggle dataset files to be available in the following directory structure:

```text
/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv
```

If you run the notebooks outside Kaggle, update the dataset paths in the configuration section accordingly.

## Models

| Model | Role |
|---|---|
| ResNet50 | Residual CNN backbone |
| VGG16 | Deep convolutional backbone |
| InceptionV3 | Inception-module CNN backbone |

The InceptionV3 notebook uses an ImageNet-pretrained backbone without its original classification head. It freezes the base layers, then unfreezes the last 30 layers for fine-tuning. A custom classification head is added:

```text
InceptionV3 backbone
        ↓
Global Average Pooling
        ↓
Dense(512) → Batch Normalization → ReLU → Dropout(0.3)
        ↓
Dense(128) → Batch Normalization → ReLU → Dropout(0.3)
        ↓
Dense(64)  → Batch Normalization → ReLU → Dropout(0.2)
        ↓
Dense(32)  → Batch Normalization → ReLU
        ↓
Dense(7) with Softmax
```

Check each model notebook for its exact backbone fine-tuning settings and classifier head. Those settings should be reported exactly as implemented.

## Pipeline

The notebooks follow this general sequence:

1. **Configure the experiment** — define image size, batch size, epoch limit, learning rate, class labels, random seed, and input/output paths.
2. **Load metadata** — read `HAM10000_metadata.csv`, locate each image in either image folder, and map diagnostic labels to integer class indices.
3. **Inspect class distribution** — count images in each class and save a distribution plot.
4. **Balance the classes** — the supplied InceptionV3 code targets 6,100 rows per class by repeating rows when a class has fewer samples and truncating a class when it has more.
5. **Create data splits** — use stratified splitting to produce 80% training, 10% validation, and 10% test subsets.
6. **Preprocess and augment images** — resize images to 224 × 224, rescale pixel values to `[0, 1]`, and apply random transformations to training images.
7. **Build the model** — load the selected pretrained CNN backbone and attach a classification head.
8. **Compile and train** — use Adamax, categorical cross-entropy, a learning-rate schedule, model checkpointing, and early stopping in the InceptionV3 notebook.
9. **Evaluate the test set** — calculate test loss and accuracy, generate predictions, and produce a classification report.
10. **Create visualisations and export artifacts** — save evaluation figures, a metrics CSV, the trained model, and a ZIP archive of figures.

## Experiment settings in the inspected InceptionV3 notebook

| Setting | Value |
|---|---|
| Input image size | `224 × 224 × 3` |
| Batch size | `32` |
| Epoch limit | `40` (as set by `EPOCHS` in code) |
| Initial learning rate | `0.001` |
| Optimizer | Adamax |
| Loss function | Categorical cross-entropy |
| Output classes | `7` |
| Random seed | `42` |
| Train / validation / test split | `80% / 10% / 10%` |
| Training augmentation | Rotation, width/height shifts, zoom, horizontal/vertical flips |
| Evaluation preprocessing | Rescaling by `1/255` |
| Model checkpoint monitor | Validation accuracy |
| Early-stopping monitor | Validation accuracy |
| Early-stopping patience | `10` epochs |
| InceptionV3 fine-tuning | Freeze backbone, then unfreeze its last 30 layers |
| Output activation | Softmax |

The source notebook's header comment mentions 30 epochs, but the executable configuration sets `EPOCHS = 40`. This README reports the value used by the code.

## Evaluation metrics and figures

The InceptionV3 notebook reports or plots:

- Test accuracy and test loss
- Per-class precision, recall, and F1-score
- Macro-average and weighted-average metrics
- Confusion matrices (raw counts and row-normalised values)
- One-vs-rest ROC curves and per-class AUC values
- Training and validation accuracy/loss curves
- Per-class prediction counts
- Prediction-confidence distribution
- Training-versus-validation accuracy gap
- Learning-rate history, if it is present in the training history

The classification report is also saved as `table_metrics.csv`.

## Output files

The notebook writes outputs to `/kaggle/working/` and creates a `thesis_figures/` subdirectory. The InceptionV3 notebook uses these main output names:

```text
/kaggle/working/
├── inceptionv3_paper1_best.h5
├── inceptionv3_paper1_final.h5
├── inceptionv3_thesis_figures.zip
└── thesis_figures/
    ├── graph01_raw_class_distribution.png
    ├── graph02_balanced_class_distribution.png
    ├── ...
    ├── graph22_macro_vs_weighted_metrics.png
    └── table_metrics.csv
```

The figure ZIP contains the figures and metrics CSV generated during execution. Exact output filenames for ResNet50 and VGG16 depend on their respective notebooks.

## Requirements

The inspected notebook uses Python with the following packages:

- TensorFlow / Keras
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn

A typical environment setup is:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

The code was written with Kaggle-style input/output paths. Package versions are not pinned in the supplied notebook, so exact reproducibility across environments is not guaranteed. For a reproducible repository, record the versions of Python, TensorFlow, NumPy, pandas, Matplotlib, seaborn, and scikit-learn used in your successful run.

## How to run

### Option A — Kaggle (matches the notebook paths)

1. Open the [HAM10000 Kaggle dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Create or open a Kaggle Notebook.
3. Add the dataset to the notebook.
4. Upload or add the relevant model notebook (`ResNet50`, `VGG16`, or `InceptionV3`).
5. Confirm that the dataset path in the notebook matches the attached Kaggle dataset location.
6. Run the notebook cells.
7. Download the exported model and figure ZIP from the notebook's output panel.

### Option B — Run in another environment

1. Download the dataset from Kaggle and extract it.
2. Update `BASE_DIR`, `IMG_DIR1`, `IMG_DIR2`, `META_PATH`, and `OUT_DIR` to valid local paths.
3. Install the dependencies listed above.
4. Run the desired notebook in Jupyter or another compatible notebook environment.

## Reproducibility and evaluation note

**Important methodological limitation in the supplied InceptionV3 code:** class balancing is performed by repeating image-metadata rows **before** the train/validation/test split. Repeated copies of the same source image can therefore be assigned to different subsets. This can cause data leakage and make test performance appear better than it would be on unseen images.

For a more reliable evaluation, split the original image records first, keeping each source image in only one subset. Apply augmentation and any oversampling **only to the training subset**. Leave validation and test data unaugmented and do not duplicate samples across those subsets. If the dataset contains multiple images from the same patient, a patient-level split is preferable when patient identifiers are available.

Also, the current notebook sets `TARGET_PER_CLASS = 6100` by repeating and truncating records; this is not the same as creating new, independent images. Describe it accurately in any manuscript.

## Repository structure

A suggested GitHub repository layout is:

```text
skin-lesion-classification/
├── README.md
├── notebooks/
│   ├── resnet50.ipynb
│   ├── vgg16.ipynb
│   └── inceptionv3.ipynb
├── results/
│   ├── resnet50/
│   ├── vgg16/
│   └── inceptionv3/
├── requirements.txt
└── .gitignore
```

Adjust the notebook filenames to match the actual names in your repository. Only commit result files you intend to share. Trained model files and generated figures can be large; consider GitHub Releases or Git LFS for large artifacts.

## Citation

Please cite the HAM10000 dataset and any research paper whose methodology or architecture you reproduce. The inspected notebook prints the following reference in its final summary:

> Villa-Pulgarin et al., *Computers, Materials & Continua* (2022), Experiment 4.

Verify the complete bibliographic details and the precise paper title from the original publication before adding a formal citation here. This README does not independently verify that reference or claim that the implementation is an exact replication.

## Disclaimer

This repository is intended for research and educational use. It is not a medical device and must not be used as a substitute for diagnosis by a qualified healthcare professional.

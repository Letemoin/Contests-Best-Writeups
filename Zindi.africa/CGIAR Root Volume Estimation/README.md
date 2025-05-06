# **CGIAR Root Volume Estimation Challenge Documentation**

## 1. Overview and Objectives

### Contest Background
Cassava is a staple food across many African countries and beyond. Accurate estimation of cassava root volume is crucial for improved crop management, selecting robust plant varieties, and addressing food security challenges. Traditional measurement methods are labor intensive and invasive. This challenge focuses on a non-destructive approach using computer vision and machine learning to estimate root volume effectively.

### Project Goals
- **Non-Invasive Estimation:** Leverage image segmentation to isolate and extract root regions from scanned images.
- **Feature Fusion:** Combine deep image features with engineered tabular data to create a robust predictive model.
- **High Accuracy:** Optimize the model to minimize RMSE in root volume estimation.
- **Scalability & Reproducibility:** Develop a modular and reproducible pipeline adaptable to future enhancements and datasets.

---

## 2. Data Description

- **Image Data:**  
  The images are stored in folders with filenames following the pattern:  
  `XXXXXXXX_S_NNN.png`  
  where `S` denotes the scan side (L for left, R for right) and `NNN` represents the layer depth. The number of images may vary between folders.

- **CSV Files:**  
  - **Train.csv** and **Test.csv** include:
    - **ID:** Unique identifier for each record.
    - **FolderName:** Name of the folder containing images.
    - **PlantNumber:** Index number of the plant in the image.
    - **Side:** Scan side (L or R).
    - **Start & End:** Recommended layer range for optimal image selection.
    - **Genotype:** Cassava variety.
    - **Stage:** Growth stage (Early or Late).
    - **RootVolume:** Target variable (only in Train.csv).

- **Segmentation Models:**  
  Three pre-trained YOLO models (`best_full.pt`, `best_early.pt`, and `best_late.pt`) are provided to detect and segment root regions in the images.

---

## 3. Architecture and Workflow

The pipeline transforms raw image and tabular data into accurate root volume predictions through a series of integrated stages.

### Overview Flowchart

```mermaid
flowchart TD
    A[Raw Image Folders & CSV Data] --> B[YOLO Segmentation]
    B --> C[Extract Root Segments]
    C --> D[Merge Segmented Images]
    D --> E[Generate Merged Image Dataset]
    A --> F[Tabular Data Processing]
    E --> G[Deep Image Feature Extraction (timm)]
    F --> H[Tabular Feature Engineering]
    G --> I[Feature Fusion (Concatenation)]
    H --> I
    I --> J[Train LightGBM Regression Model]
    J --> K[Model Evaluation (RMSE)]
    K --> L[Predict Root Volume on Test Data]
    L --> M[Submission File Creation]
```

## 4. Detailed Steps

### 4.1 Preprocessing and Data Preparation

**Library Imports & Environment Setup:**  
- Import essential libraries such as NumPy, Pandas, Matplotlib, Pillow, OpenCV, PyTorch, YOLO (ultralytics), timm, scikit-learn, PyTorch Lightning, and LightGBM.

**Seed Initialization:**  
- Ensure reproducibility using a dedicated seed initialization function.

**Path and Variable Configuration:**  
- Set up file paths (e.g., `TRAIN_DATA_PATH`, `TEST_DATA_PATH`).
- Define image dimensions, batch sizes, and specify model parameters.

### 4.2 Image Segmentation and Merging

**Loading Segmentation Models:**  
- Load the three YOLO segmentation models corresponding to the early, late, and full growth stages.

**Root Region Detection:**  
- Implement `get_segmented_images` to apply the YOLO models and extract root segments from each image.

**Image Merging:**  
- Use `merge_segmented_images` to combine segmented regions across the specified layer range, creating a single merged image per sample.

**Dataset Generation:**  
- Iterate through each sample in the CSV files to generate new CSVs with file paths pointing to the merged images.

### 4.3 Feature Extraction

**Deep Image Features:**  
- Employ a timm pretrained model (e.g., `focalnet_xlarge_fl3.ms_in22k`) to extract high-dimensional feature vectors from the merged images, capturing detailed spatial information.

**Tabular Feature Engineering:**  
- Process metadata by:
  - Converting numerical columns (e.g., `PlantNumber`) to proper numeric types.
  - Applying one-hot encoding to categorical variables such as `Side`, `Genotype`, and `Stage`.

**Feature Fusion:**  
- Concatenate the image features with the engineered tabular features to form a comprehensive feature set for each sample.

### 4.4 Model Training and Evaluation

**Training Setup:**  
- Split the processed data into training and validation sets.

**LightGBM Regression:**  
- Train a LightGBM model on the fused feature set using techniques like early stopping to optimize performance and prevent overfitting.

**Performance Assessment:**  
- Evaluate the model using RMSE on both the training and validation sets.

**Predictions:**  
- Generate root volume predictions for the test set and incorporate these into the test DataFrame.

### 4.5 Inference and Submission

**Inference Pipeline:**  
- Deploy the trained model on unseen data to produce predictions for root volume.

**Submission File Creation:**  
- Assemble a final CSV file containing two columns (`ID` and `RootVolume`) to be submitted for evaluation.

## 5. Error Handling and Logging

**Data Integrity:**  
- Validate the availability and proper naming of image files prior to processing.

**Segmentation Exceptions:**  
- Log instances where no detections occur and gracefully handle exceptions during image merging.

**Training Monitoring:**  
- Use callbacks (e.g., EarlyStopping, ModelCheckpoint) to continuously monitor model training and mitigate overfitting.

**Inference Logging:**  
- Maintain detailed logs during batch inference to capture any discrepancies or errors in data processing.

## 6. Conclusion

This innovative pipeline integrates non-invasive image segmentation with deep feature extraction and robust tabular data processing to accurately estimate cassava root volume. By leveraging state-of-the-art techniques and ensuring reproducibility, the solution provides a scalable framework for enhanced crop management and agricultural decision-making. The detailed documentation outlines a clear, modular approach from raw data ingestion to final prediction submission, laying the foundation for future enhancements and research in agricultural technology.






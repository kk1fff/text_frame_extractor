# Multi-Frame Page Reconstruction and Understanding System

## Objective

Design and implement a system that processes multiple video frames of a reading surface (such as a book, tablet, newspaper, etc.), reconstructs a clean, flat, and unobstructed image of the page, and extracts its text content while preserving paragraph structure.

The system must handle challenges such as:

* The target region appearing at different positions, scales, and orientations in each frame
* The target region being tilted, warped, or curved
* Frames with occlusions (e.g., fingers, hands, glare)
* Frames that are blurry or redundant

---

## Key Concept: Target Region Detection

**Definition:**

> Target Region Detection is the process of identifying and localizing the **region of interest (ROI)** in each frame that contains the reading material.
> This region may correspond to a physical book, a tablet screen, an e-reader, a newspaper, or any other surface with readable content.

The detected target region is used as the basis for alignment, reconstruction, and text extraction.

---

## Pipeline Overview

### High-Level Stages

1. Frame Selection
2. Target Region Detection
3. Frame Registration and Perspective Correction
4. Occlusion Detection and Masking
5. Page Reconstruction (Fusion)
6. Optional: Dewarping
7. OCR
8. Text Structure Analysis
9. Quality Scoring
10. Output

---

## Detailed Components

### 1. Frame Selection

**Purpose:** Select the most informative and complementary frames from all available candidates.

**Criteria:**

* Sharpness (focus, using Laplacian variance)
* Minimal occlusion (low hand/finger coverage)
* Coverage gain (uncovering new regions)
* Alignment error (lower residuals preferred)
* Diversity (complementary viewpoints, avoid redundancy)

**Output:** Top-N frames ranked and selected.

---

### 2. Target Region Detection

**Purpose:** Detect and localize the **target reading region** in each frame, even if it appears at different positions, scales, or orientations.

**Techniques:**

* Object detection (e.g., YOLO with a `document` or `device` class) → bounding box
* Contour detection (largest quadrilateral in frame)
* Feature matching to a reference template

**Output:** Target region bounding box or corner coordinates (quadrilateral) per frame.

---

### 3. Frame Registration and Perspective Correction

**Purpose:** Align each detected target region to a common reference coordinate system.

**Techniques:**

* Estimate homography between detected region corners and reference rectangle
* Warp each frame to align with the reference plane (`cv2.warpPerspective`)

**Output:** All frames geometrically aligned in the same target space.

---

### 4. Occlusion Detection and Masking

**Purpose:** Identify and mask obstructed regions (hands, fingers, glare) in each aligned frame.

**Techniques:**

* Skin detection (color thresholds)
* Semantic segmentation (trained model)
* Background subtraction

**Output:** Validity mask for each aligned frame.

---

### 5. Page Reconstruction (Fusion)

**Purpose:** Combine aligned, masked frames into a single clean, complete image of the target region.

**Techniques:**

* Per-pixel blending:

  * Median blending
  * Weighted averaging
  * Highest-confidence pixel selection
* Filling in previously occluded regions using unobstructed frames

**Output:** Reconstructed target region image.

---

### 6. Optional: Dewarping

**Purpose:** Flatten curved or warped pages for better readability and OCR.

**Techniques:**

* Contour detection and geometric correction
* Thin Plate Splines (TPS)
* Deep learning models (e.g., DocUNet, DewarpNet)

**Output:** Flat, rectangular page image.

---

### 7. OCR

**Purpose:** Extract text and bounding boxes from the reconstructed target region image.

**Techniques:**

* Tesseract OCR
* PaddleOCR
* EasyOCR or GPT-4 Vision

**Output:** Text content with bounding boxes and confidence scores.

---

### 8. Text Structure Analysis

**Purpose:** Organize recognized text into meaningful lines, paragraphs, and headers.

**Techniques:**

* Group word boxes into lines based on proximity and alignment
* Merge lines into paragraphs based on spacing
* Detect headers, lists, tables, or columns if applicable

**Output:** Structured text with logical hierarchy.

---

### 9. Quality Scoring

**Purpose:** Evaluate the quality of the reconstructed region and the extracted text.

**Metrics:**

* Coverage of the target region (text area over region area)
* Sharpness (focus on text regions)
* OCR confidence (mean/median)
* Line alignment and straightness
* Paragraph coherence

**Output:** Quality score and optional pass/fail recommendation.

---

### 10. Output

**Purpose:** Deliver final results for downstream use.

**Outputs:**

* Clean, reconstructed target region image
* OCR text with bounding boxes
* Structured paragraphs and hierarchy
* Quality score

---

## Non-Feature Requirements

These are quality, operational, and usability requirements that the system must satisfy but are not part of its core functionality.

### Real-Time Feedback

* The system should provide a real-time or near-real-time preview of the reconstruction progress and quality, allowing the user to adjust camera position or capture more frames if needed.

### Debugging and Visualization

* The system should offer optional visual overlays for debugging and analysis:

  * Detected target region bounding boxes or contours
  * Masks showing detected occlusions
  * Heatmaps illustrating text coverage and sharpness
  * Aligned and blended frames at each stage

### Robustness and Fallback

* If the reconstruction quality falls below a defined threshold (e.g., insufficient coverage or low OCR confidence), the system should:

  * Warn the user
  * Suggest re-capturing frames
  * Optionally retry with different parameters or fewer/more frames

### Maintainability and Extensibility

* The pipeline should be modular, with clearly defined interfaces between stages, so that components can be updated or replaced (e.g., switching from Tesseract to PaddleOCR) without requiring major rewrites.

### Performance

* The system should be optimized to run on a standard PC in a reasonable time (e.g., under 10 seconds per page) while supporting scalability to higher-resolution images and more frames if needed.

---

## Summary Table

| Stage                                   | Input                    | Output                                |
| --------------------------------------- | ------------------------ | ------------------------------------- |
| Frame Selection                         | Raw frames               | Top-N frames                          |
| Target Region Detection                 | Selected frame           | Target region bounding box or corners |
| Registration and Perspective Correction | Target regions in frames | Aligned target region images          |
| Occlusion Detection and Masking         | Aligned frames           | Validity masks per frame              |
| Page Reconstruction                     | Frames + masks           | Clean, fused target region image      |
| Dewarping (optional)                    | Fused image              | Flattened target region image         |
| OCR                                     | Reconstructed image      | Text with bounding boxes              |
| Text Structure Analysis                 | OCR output               | Paragraphs, lines, hierarchy          |
| Quality Scoring                         | All outputs              | Quality score                         |
| Final Output                            | Reconstructed + text     | Ready-to-use content                  |


## Running the Example

Install dependencies:

```bash
pip install -r requirements.txt
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

Run the pipeline on a video file:

```bash
python main.py input.mp4 output.jpg
```

The reconstructed image will be saved to `output.jpg` and extracted text and quality score will be printed to the console.


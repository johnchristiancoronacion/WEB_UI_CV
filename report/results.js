window.labResults = {
  1: {
    summary: "The street sample measured 810 × 1080 px. At (405, 540), the pixel changed from blue (155, 85, 2 BGR) to magenta (255, 0, 255 BGR), as shown in the enlarged before-and-after view.",
    metrics: {
      "Array shape (height, width, channels)": "(1080, 810, 3)",
      "Grayscale mean intensity": "116.43",
      "Crop origin (x, y)": "(142, 189)",
      "Crop dimensions": "526 × 702 px",
      "Resized dimensions": "526 × 702 px",
      "Rotated canvas": "1080 × 810 px",
    },
    conclusion: "The edit changed a single location without changing image dimensions. The 90° turn exchanged width and height. The crop selected a smaller region of the source image, while resizing resampled the entire image to the same output size.",
  },
  2: {
    summary: "The street image was smoothed with a 3 × 3 Gaussian kernel. With thresholds of 100 and 200, Canny detected 96,283 edge pixels, or 11.01% of the image.",
    metrics: {
      "Gaussian kernel": "3 × 3",
      "Canny low / high": "100 / 200",
      "Peak Sobel magnitude": "695.09",
      "Canny edge pixels": "96,283",
      "Canny edge density": "11.01%",
    },
    conclusion: "Sobel preserves graded responses along the street scene’s intensity changes. Canny traces thin binary outlines along strong transitions, including the bus boundary and the outlines of pedestrians, while many weaker responses are absent from the binary map.",
  },
  3: {
    summary: "Global equalization gave the widest measured intensity spread on the street sample. Linear stretching and CLAHE both increased the grayscale standard deviation from its original 32.66.",
    metrics: {
      "Original standard deviation": "32.66",
      "Global equalization standard deviation": "73.74",
      "Linear stretch standard deviation": "68.27",
      "CLAHE standard deviation": "54.90",
      "CLAHE clip limit": "2",
      "CLAHE tile grid": "8 × 8",
    },
    conclusion: "Global equalization produced the largest overall increase in contrast on this sample. CLAHE enhanced contrast within local tiles with a smaller global spread. Global equalization can also amplify noise in flat areas, while a CLAHE clip limit restrains local amplification.",
  },
  4: {
    summary: "The comparison used the street sample and a copy rotated 12° and scaled to 90%. Oriented SURF found 81 matches with a 1% Hessian threshold. Upright SURF found 61 matches in a separate run of the same pair.",
    metrics: {
      "SURF detected / displayed / matched": "3,261 / 150 / 81",
      "SIFT displayed / matched": "150 / 80",
      "ORB displayed / matched": "150 / 65",
      "SURF extraction for both images": "2,411.3 ms",
      "SIFT extraction for both images": "261.1 ms",
      "ORB extraction for both images": "133.0 ms",
    },
    conclusion: "SURF gave the most matches in this trial and ORB extracted descriptors fastest. Assigning orientation helped SURF match the rotated sample. Raising the Hessian threshold generally removes weaker keypoints. Descriptors encode local appearance for matching between views.",
  },
  5: {
    summary: "The 1,288 LFW images represent seven identities. The 150 whitened components kept 94.56% of the training variance, and the cumulative variance curve reached 90% at 85 components. The SVM misclassified 39 of the 258 test faces.",
    metrics: {
      "Features per face": "1,850 (37 × 50 px)",
      "Training / test images": "1,030 / 258",
      "Components for 90% variance": "85",
      "Variance retained at 150": "94.56%",
      "Test accuracy": "84.88%",
      "Macro F1": "0.800",
    },
  },
  6: {
    summary: "Pretrained YOLO11n detected one bus and four people on the street sample. At 25% minimum confidence, the fine-tuned model detected the orange badge (99.19%) and blue card (99.17%) in the held-out card image.",
    metrics: {
      "Pretrained bus confidence": "93.92%",
      "Pretrained person confidence range": "39.93–90.20%",
      "Validation precision / recall": "99.22% / 100%",
      "Validation mAP50 / mAP50–95": "99.50% / 92.06%",
      "Held-out test precision / recall": "99.00% / 100%",
      "Held-out test mAP50 / mAP50–95": "99.50% / 89.17%",
    },
  },
  7: {
    summary: "On the street sample, adaptive thresholding selected more area than Otsu. Seeded region growing selected a local connected area from (405, 540) with tolerance 20. Coverage describes the mask area for this unannotated input.",
    metrics: {
      "Otsu threshold / coverage": "117 / 49.05%",
      "Adaptive threshold coverage": "62.44%",
      "Seeded region coverage": "1.87%",
      "Watershed seed regions": "6",
      "Canny contours / coverage": "6 / 66.33%",
      "Semantic classes / largest": "16 / Road (28.3%)",
    },
  },
};

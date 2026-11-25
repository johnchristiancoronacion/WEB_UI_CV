# CSST106 Computer Vision Laboratory

This Flask app organizes seven CSST106 laboratory activities in one browser interface. Select a lab, run its sample or upload suitable input, and open the resulting images to inspect them at full size or download them.

## Run on Windows

With Python 3.12 installed, run these commands in PowerShell from the project root. The three download commands cache the external dataset and pretrained models for Labs 5–7.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py --download-lfw
.\.venv\Scripts\python.exe app.py --download-model
.\.venv\Scripts\python.exe app.py --download-segmentation
.\.venv\Scripts\python.exe app.py
```

Open [Vision Lab locally](http://127.0.0.1:8000). Use `--port 8001` when starting the app if port 8000 is in use.

## Lab activities

| Activity | What to review |
| --- | --- |
| 01 · Image Manipulation | Inspect image shape and BGR values. Edit a pixel, view the enlarged change, and compare grayscale, channels, resizing, rotation, flipping, and region cropping. |
| 02 · Edge Detection | Compare the original grayscale image, Sobel magnitude, and Canny in one view. Adjust the Gaussian kernel and Canny thresholds, then compare the other gradient and edge maps. |
| 03 · Image Histograms | Examine RGB and grayscale histograms and compare global equalization, CLAHE, linear contrast stretching, and color equalization on the YUV luminance channel. |
| 04 · SURF Feature Detection | Compare SURF keypoints and descriptor matches with SIFT and ORB using two uploaded images or a rotated sample. Toggle upright SURF and compare matches. |
| 05 · PCA and Eigenfaces | Review 150 whitened principal components, the cumulative variance plot, and a balanced RBF SVM classification report on a held-out LFW split. The sample run classified 84.88% of the 258 test faces correctly. The image reconstruction mode accepts an uploaded image. |
| 06 · YOLO11 Object Detection | Run the pretrained COCO model or a custom card-and-badge model on images, short video, or the webcam. The training panel shows held-out precision, recall, and mAP. The Lab 6 exercise specifies YOLO11 despite its attachment filename. |
| 07 · Image Segmentation | Compare Otsu and adaptive thresholding, seeded region growing, watershed, Canny contours, K-means, and SegFormer pixel labels. The semantic example shows a Cityscapes class map and its pixel shares. |

Image uploads support JPEG, PNG, BMP, and WebP with an 8-megapixel limit. Requests have a 25 MB limit. Lab 6 samples at up to four frames per second from the first 12 seconds of a supported video and produces an annotated silent WebM file. Webcam capture requires browser permission and works on localhost.

## Reproduce Lab 6 training

The bundled model was fine-tuned on a reproducible synthetic dataset of blue cards and orange badges. The Python script prepares the images, normalized YOLO annotations, and dataset YAML, then trains, validates, tests, and exports `models/lab6-custom.onnx`. The five-epoch 192 × 192 run achieved 89.17% mAP50–95 on the 16-image synthetic test split.

```powershell
.\.venv\Scripts\python.exe -m pip install -r training/requirements.txt
.\.venv\Scripts\python.exe -m training.train
```

Use `--dataset-only` to generate the annotated dataset alone or `--epochs 100 --imgsz 640` to run the settings illustrated in the handout. [Recorded metrics](training/results.json) · [Custom inference screenshot](report/screenshots/lab-06-custom.png)

## Laboratory report

The eight-page A4 report includes objectives, procedures, techniques, measurements, conclusions, and screenshots from the Web UI. Fill in the cover details, then print the PDF for submission. [Printable report](report/CSST106-Laboratory-Report.pdf) · [HTML report](report/index.html)

The application entry point is `app.py`. The browser interface is under `web/`, the processing modules are `vision_*.py`, and the training workflow is under `training/`.

## References and assets

- The included street scene is the Ultralytics demo photo. [Original bus.jpg](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg)
- The pretrained detector uses the Ultralytics YOLO11n COCO release. The included custom detector was fine-tuned from the same model family. [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) · [Release assets](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
- Bay, Ess, Tuytelaars, and Van Gool describe the Fast-Hessian detector, Haar-wavelet orientation, and SURF descriptor. [Speeded-Up Robust Features](https://doi.org/10.1016/j.cviu.2007.09.014)
- The eigenfaces exercise uses the scikit-learn LFW loader and follows the handout’s minimum of 70 photos per identity. [LFW dataset](https://vis-www.cs.umass.edu/lfw/) · [scikit-learn LFW loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)
- The semantic segmentation exercise uses NVIDIA’s SegFormer-B0 fine-tuned on Cityscapes and Xenova’s ONNX conversion. [Original model](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-512-1024) · [ONNX export](https://huggingface.co/Xenova/segformer-b0-finetuned-cityscapes-512-1024)

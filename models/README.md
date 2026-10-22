# Vision Guard — Models Directory

This directory stores deep learning weights and model assets utilized by the Vision Guard Computer Vision pipeline.

## 1. Pretrained Base Model (`yolov8n.pt`)
* **Framework:** Ultralytics YOLOv8 (Nano architecture)
* **Status:** Installed & verified in this project.
* **Trained on:** COCO Dataset (80 object classes).
* **Supported Traffic Classes:**
  * Class ID 0: `person`
  * Class ID 1: `bicycle`
  * Class ID 2: `car`
  * Class ID 3: `motorcycle`
  * Class ID 5: `bus`
  * Class ID 7: `truck`
  * Class ID 9: `traffic light`

---

## 2. Specialized Helmet Detection Model (`helmet_yolo.pt`)
* **Standard Behavior:**
  * Generic YOLO COCO weights do **NOT** have a dedicated `helmet` class. Generic models only detect `motorcycle` and `person`.
  * Vision Guard features an intelligent dual-mode engine:
    1. If `helmet_yolo.pt` is present, it directly runs deep inference for classes `[helmet, no_helmet]`.
    2. If `helmet_yolo.pt` is not present, Vision Guard falls back to its Computer Vision geometric & head-region morphological feature analyzer (evaluating cranial contour curvature, aspect ratio, skin/hair tone vs shell sheen) and categorizes uncertain visual evidence for manual review.
* **Recommended Datasets for Fine-Tuning:**
  * **Kaggle Motorcycle Helmet Detection Dataset:** [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)
  * **Roboflow Universe Helmet Detection:** [Roboflow Universe](https://universe.roboflow.com/search?q=helmet+detection)
* **Training Command:**
  ```bash
  yolo task=detect mode=train model=yolov8n.pt data=helmet_dataset.yaml epochs=50 imgsz=640
  ```

---

## 3. Seat-belt Compliance Model (`seatbelt_yolo.pt`)
* **Standard Behavior:**
  * General COCO weights cannot directly predict seat-belts inside a vehicle cabin through windshield glare.
  * Vision Guard applies vehicle occupant cabin localization (upper windshield zone) combined with multi-angle diagonal Sobel/Canny edge filtering and Hough Line intensity profiling.
  * If a custom trained `seatbelt_yolo.pt` is placed here, the pipeline seamlessly switches to deep bounding box prediction.
* **Recommended Datasets:**
  * **Roboflow Driver Seatbelt Detection:** [Roboflow Universe Seatbelt](https://universe.roboflow.com/search?q=seatbelt)
  * **AutoData Traffic Safety Benchmark:** [OpenDataLab](https://opendatalab.com/)

---

## 4. Number Plate OCR Engine
* **Engine:** EasyOCR with PyTorch backend & OpenCV adaptive morphological preprocessing.
* **Character Set:** Latin Alphanumeric (`en`).
* **Localization:** Bounding-box aspect ratio thresholding (2.5:1 to 5.5:1) + adaptive thresholding.
* **Model Cache:** Downloaded automatically to `~/.EasyOCR/model/` on first execution.

\# Evaluation Summary



\## Model



\- Model architecture: SSD MobileNet V2 FPNLite 320x320

\- Framework: TensorFlow Object Detection API

\- Training approach: Transfer learning

\- Number of classes: 5

\- Classes: hello, thanks, yes, no, iloveyou



\## Dataset



\- Dataset type: Custom webcam dataset

\- Annotation format: Pascal VOC XML

\- Training format: TFRecord

\- Split: Train/Test



\## Training



\- Training steps: 2000

\- Final total loss: approximately 0.29

\- Final classification loss: approximately 0.10

\- Final localization loss: approximately 0.04



\## Real-Time Inference



\- Webcam-based detection using OpenCV

\- Confidence score displayed

\- Temporal prediction smoothing added

\- Sentence builder added

\- Text-to-speech output added



\## Notes



The model performs well on the initial custom dataset but can be improved with more signs, cleaner images, stronger lighting variation, and more diverse backgrounds.


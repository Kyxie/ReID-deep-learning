## Re-ID Deep Learning

This is my Final Year Project in University of Electronic Science and Technology of China @ 2021.

### Features

- This project mainly focus on Pedestrian Re-Identification based on Deep Learning Methods.
- We used Market-1501 Dataset to train the model and used our own made dataset, UESTC Re-ID Dataset, and Market-1501 to test the model.
- The model is based on ResNet50 with TriHard Loss. 
- We trained 60 epochs.
- Market-1501 test results:
  - mAP: 58.8%
  - rank@1: 76.3%
  - rank@5: 90.1%
- UESTC Re-ID Dataset test results:
  - mAP: 74.6%
  - rank@1: 80.0%
  - rank@5: 93.3%

### Codes

- Train file ----- ResNet_metric.py
- Test file ----- tets_UESTC.py, test_market1501.py

### Acknowledgement

[Dr. Hao Luo (michuanhaohao)](https://github.com/michuanhaohao)

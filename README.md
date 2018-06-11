# PyTorch implementation of Luo et. al.: Efficient Deep Learning for Stereo Matching

Requires PyTorch 0.4

To train:
- Download the Kitti stereo dataset
- Edit KITTIPATH in config.py
- Edit AND create the save folder in config.py
- Run train.py
A model is saved after every epoch

To test:
- Edit image paths in test.py
- Edit TEST_EPOCH in config.py to select the model to test
- Run test.py

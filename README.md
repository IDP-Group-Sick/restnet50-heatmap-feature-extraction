# Extract Image Features using ResNet50
From the pre-trained [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50) model, the last two layers will be removed so it can be used to create features from images.

The file [extract-features.py](extract-features.py) showcases how heatmap `.png`-files can be loaded, scaled and fed through this modified ResNet50.

The images that are loaded can also be substituted by any representation of `numpy`-arrays. So one can create them in another workflow and use this code as a guide to create the predictions.
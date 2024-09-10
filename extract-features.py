# %%
import os
import glob
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50

# %% load heatmaps
heatmap_directory = os.path.normpath("/host-homes/bule/explainable_AI/pictures/10Fold_CIB/")
#heatmap_file = "all_ensemble_heatmaps_CIB_M2_wgt_gc_predcl_unnormalized_co0.995.npy"
heatmap_file = "all_max_activation_indices_CIB_M2_wgt_oc_predcl_unnormalized.npy"

# Available Heatmap files:
# ll -t /home/bule/explainable_AI/pictures/10Fold_CIB/ | grep npy
# -rw-r--r-- 1 bule bule 1642358648 Jul  7 17:35 all_ensemble_heatmaps_CIB_M2_wgt_gc_predcl_unnormalized_co0.995 npy
# -rw-r--r-- 1 bule bule 1493696640 Jul  7 17:35 all_heatmaps_CIB_M2_wgt_gc_predcl_unnormalized_co0.995.npy
# -rw-r--r-- 1 bule bule 1642358648 Mär  5  2024 all_ensemble_heatmaps_CIB_M2_wgt_oc_predcl_unnormalized.npy
# -rw-r--r-- 1 bule bule 1493696640 Mär  5  2024 all_heatmaps_CIB_M2_wgt_oc_predcl_unnormalized.npy
# -rw-r--r-- 1 bule bule 1642358648 Feb 29  2024 all_ensemble_heatmaps_CIB_M2_wgt_gc_predcl_unnormalized.npy
# -rw-r--r-- 1 bule bule 1493696640 Feb 29  2024 all_heatmaps_CIB_M2_wgt_gc_predcl_unnormalized.npy
# ...

heatmaps = np.load(os.path.join(heatmap_directory, heatmap_file), allow_pickle=True)
print(f"Length of heatmap list: {len(heatmaps)}")
print(heatmaps[0])

# HEATMAPS ARE USELESS -> WE NEED THE IMAGES (LAYER THAT WE HAVE LOOKED AT)
# Alternatively we need to find the logic that generates the images and pull the numpy arrays from there.

########### Generally:
# We want to create a numpy array that contains images with the input dimension of the network
# for resnet 50 it is (244, 244, 3)
# Here we are reading the generated images but we can also take other "images", as long 
# as the input dimension is guaranteed.

# %% loading the images into a list
heatmap_image_filter = "CIB_M2_wgt_gc_predcl_unnormalized_co0.995_last_layer_avg_max_orig.png"
heatmap_image_filter_with_wildcard = f"*{heatmap_image_filter}"
# As we see further below, our resnet model expects (244, 244, 3)
resize_size = (224, 224, 3)

images = []
for image_path in glob.glob(os.path.join(heatmap_directory, heatmap_image_filter_with_wildcard)):
    image_frame = Image.open(image_path)
    background = Image.new('RGBA', image_frame.size, (255,255,255))
    alpha_composite = Image.alpha_composite(background, image_frame)
    alpha_composite_3 = alpha_composite.convert('RGB')

    resized_image = alpha_composite_3.resize(resize_size[0:2])

    np_frame = np.array(resized_image)
    print(np_frame.shape)
    images.append(np_frame)

# Cast the list of arrays to an array
images_array = np.stack(images)
print(f"Size of image array: {images_array.shape}")

# %% load resnet50
resnet_model = ResNet50()
#print(model.summary())
resnet_model_names = [layer.name for layer in resnet_model.layers[-10:]]
print("Last 10 layers of resnet50:", *resnet_model_names, sep="\n")
print("So we pop the last two layers...")

# %% remove layers
resnet_model._layers.pop() #predictions
resnet_model._layers.pop() #avg_pool

# %% inspect new model
modified_resnet_model_names = [layer.name for layer in resnet_model.layers[-10:]]
print("Last 10 layers of the modified network:", *modified_resnet_model_names, sep="\n")

# %% resize images to resnet input dimensions
input_shape = resnet_model.layers[0].input_shape[0]
input_shape # size is (None, 224, 224, 3)

assert resize_size == input_shape[1:], "The size that images were scaled to do not line up with the loaded models input dimension."


# %% Predict/generate the features
predictions = resnet_model.predict(images_array, verbose=0)

# %% Save the predictions
np.save("./generated-features.npy", predictions)
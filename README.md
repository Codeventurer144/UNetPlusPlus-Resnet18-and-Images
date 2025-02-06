# ResNet-18 and U-Net++

## Description:

In this pytorch project, I used performed **transfer learning** on an **Ants and Bees dataset**  making use of **ResNet-18**. I also trained a **U-Net++** for **image segmentation**. For the Resnet-18 Model, I used **torchvision.transforms**, I applied different **image augmentation techniques** and normalized the images using their **mean** and **standard deviation**. I defined a custom function, **def** `train_model()`, to streamline the model training process and to facilitate the usage of the **ResNet-18 architecture** for both **forward propagation** and **backpropagation**, as well as calculating and monitoring its  **loss function and accuracy**. The **classification accuracy** achieved with the **ResNet-18** model was **94%**.

As for U-net++, which is my focus in this repository, I encountered several Issues such as the kind of U-net++ Architecture I was using, the incompatibility of Libraries, the still-unknown reason for it not running on my Laptop but on github codespaces😅, and Skip connections receiving different dimensions of inputs than they expected. I thanfully found a repository on github that helped me to a great extent (though I still need to do some debugging) to train the model.

`preprocess_dsb2018.py` - This script processes the image and mask data from the provided dataset, preparing them for training the model. It loads images and associated mask data, resizes them to a specified size, converts mask images to binary format where mask pixels are marked as 1 and the rest as 0, and saves the processed images and masks into organized directories, ready for model input.

`dataset.py` - This script defined a custom dataset class for PyTorch to use which was intended to handle image and mask pairs for tasks like image segmentation. Here's a brief breakdown of what it does:

1. **Initialization (`__init__`)**:
   - The dataset is initialized with:
     - `img_ids`: A list of image identifiers.
     - `img_dir`: The directory containing the image files.
     - `mask_dir`: The directory containing the mask files.
     - `img_ext` and `mask_ext`: Extensions for image and mask files (e.g., `.jpg`, `.png`).
     - `num_classes`: The number of mask classes (e.g., background, object, etc.).
     - `transform`: Optional image and mask transformations (e.g., augmentation) to apply during data loading.
   - The `img_ids` are expected to correspond to both images and their associated masks.

2. **Length (`__len__`)**:
   - This method returns the number of images in the dataset by returning the length of `img_ids`.

3. **Item Retrieval (`__getitem__`)**:
   - This method retrieves an image and its corresponding mask given an index (`idx`).
   - It:
     - Loads the image using OpenCV (`cv2.imread`).
     - Loads masks for each class (the mask is read as grayscale and stacked together into a multi-channel array for all classes).
     - Optionally applies any transformations to the image and masks using the `transform` parameter (e.g., from `albumentations` library).
     - Normalizes the image and mask (divides by 255 for scaling between 0 and 1).
     - Reorders the dimensions of the image and mask (to channel-first format: `C x H x W`).
     - Returns the processed image, mask, and a dictionary containing the image ID.


### Key Features:
- The dataset expects images and masks to be stored in a specific folder structure, with each mask class stored in separate subfolders (one folder for each class).
- It supports transformations (e.g., for data augmentation) applied during runtime to the images and masks.
- It outputs images and masks in a format suitable for PyTorch models, with images scaled and masks stacked into multiple channels.

This setup is common for segmentation tasks where each image has multiple associated masks representing different object classes or categories.

-----------------------

Currently, I am learning about the **U-Net++ architecture**—an evolution from the **U-Net**—which introduces **dense skip pathways** between the encoder and decoder blocks. These dense skip connections significantly improve **feature propagation**, enhance **gradient flow**, and allow for the capture of more **fine-grained details** in the segmentation output. Unlike U-Net, **U-Net++** creates a series of **nested skip connections** at each level, bridging the encoder and decoder at multiple levels. This design ensures that the decoder receives rich features from multiple encoder layers, helping to improve **localization accuracy** and **contextual information** during the image segmentation process.

![](U-Net.jpg)

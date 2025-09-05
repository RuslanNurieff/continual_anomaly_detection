import torch
import torchvision.transforms as transforms
import os
import random
import io
import numpy as np
import argparse
from PIL import Image

from moviad.datasets.bmad.bmad_dataset import BMAD
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.configurations import TaskType
from moviad.models.patchcore.product_quantizer import ProductQuantizer


def compress_features(dataset_path, categories, device, backbone, layer_idxs, compression_method, 
                      quality, feature_dtype, pq_method, pq_subspaces, centroids_per_subspace):
    
    print(f"Compressing images using {compression_method} with quality of {quality}")
    print(f"Casting features to {feature_dtype}")

    for category in categories:
        print(f"Compressing {category} category")

        #load dataset
        train_dataset = BMAD(TaskType.SEGMENTATION, dataset_path, category, "train")
        dataset_images = train_dataset.samples.sample(len(train_dataset))
        print(f"Number of images to compress: {len(train_dataset)}")
    
        feature_extractor = CustomFeatureExtractor(backbone, layer_idxs, device = device)

        if pq_method == "layerwise":
            feature_vectors = {idx: [] for idx in layer_idxs}
            trained_quantizers = {}
        
        #transform images to be passed to the backbone
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        original_sizes = []
        compressed_sizes = []
        feature_sizes = []
        pq_feature_sizes = []

        features_list = [] #list for global product quantizer

        #collect features
        print("Extracting and storing features...")
        for _, sample in dataset_images.iterrows():
            image_path = sample["image_path"]
            original_image = Image.open(image_path).convert("RGB")

            #apply transformations
            image_tensor = transform(original_image).unsqueeze(0).to(device)

            #feature extraction
            features = feature_extractor(image_tensor) #list of [1, C, H, W]
            feature_tensor = torch.cat([f.flatten() for f in features])
            features_list.append(feature_tensor.cpu())

            if pq_method == "layerwise":
                for layer_idx, feature_map in zip(layer_idxs, features):
                    layer_feature_tensor = feature_map.flatten()
                    feature_vectors[layer_idx].append(layer_feature_tensor)

        #train product quantizers
        if pq_method == "global":
            pq = ProductQuantizer(subspaces = pq_subspaces, centroids_per_subspace= centroids_per_subspace)
            feature_matrix = torch.stack(features_list)  
            print(f"Training global PQ with {feature_matrix.shape[0]} vectors...")
            pq.fit(feature_matrix)  
        elif pq_method == "layerwise":
            for idx, vectors in feature_vectors.items():
                print(f"Training PQ for layer {idx} with {len(vectors)} vectors...")
                X = np.stack(vectors)
                layer_pq = ProductQuantizer(subspaces = pq_subspaces, centroids_per_subspace = centroids_per_subspace)
                layer_pq.fit(X)
                trained_quantizers[idx] = layer_pq

        #encode and measure sizes
        print("Encoding features and measuring sizes...")
        for _, sample in dataset_images.iterrows():
            image_path = sample["image_path"]
            original_image = Image.open(image_path).convert("RGB")

            #compute the original size
            original_size = os.path.getsize(image_path)
            original_sizes.append(original_size)

            #compress image
            compressed_image_io = io.BytesIO()
            original_image.save(compressed_image_io, format=compression_method, quality = quality) 
            compressed_size = compressed_image_io.tell()
            compressed_sizes.append(compressed_size)

            #compute feature size
            features = feature_extractor(image_tensor) #list of [1, C, H, W]
            feature_tensor = torch.cat([f.flatten() for f in features])
            feature_tensor = feature_tensor.to(dtype = feature_dtype)
            feature_size = feature_tensor.numel() * feature_tensor.element_size()
            feature_sizes.append(feature_size)

            #encode and measure size of compressed features
            if pq_method == "global":
                compressed_feature = pq.encode(feature_tensor.unsqueeze(0)) 
                pq_feature_size = compressed_feature.numel() * compressed_feature.element_size()
                pq_feature_sizes.append(pq_feature_size)
            elif pq_method == "layerwise":
                pq_size = 0
                for layer_idx, feature_map in zip(layer_idxs, features):
                    layer_pq = trained_quantizers[layer_idx]
                    layer_feature_tensor = feature_map.flatten()
                    layer_feature_tensor = layer_feature_tensor.unsqueeze(0)
                    encoded = layer_pq.encode(layer_feature_tensor) #[1, M]
                    pq_size += encoded.numel() * encoded.element_size()
                pq_feature_sizes.append(pq_size)
        
        # Compute averages
        avg_original_size = np.mean(original_sizes)
        avg_compressed_size = np.mean(compressed_sizes)
        avg_feature_size = np.mean(feature_sizes) 
        avg_pq_feature_size = np.mean(pq_feature_sizes) 

        # Print results
        #print(f"Average original size: {int(avg_original_size)} bytes")
        #print(f"Average compressed size: {int(avg_compressed_size)} bytes")
        print(f"Compression ratio of images: {1 - (sum(compressed_sizes) / sum(original_sizes)): .2%}")

        print(f"Average feature size: {int(avg_feature_size)} bytes")
        print(f"Compression ratio of features/images: {1 - (sum(feature_sizes) / sum(original_sizes)): .2%}")

        if pq_method is not None:
            print(f"Average PQ feature size: {int(avg_pq_feature_size)} bytes")
            print(f"Compression ratio using PQ features: {1 - (sum(pq_feature_sizes) / sum(feature_sizes)): .2%}")

    return




def main():

    categories = ["brain", "chest", "histopathology", "liver", "retinaoct", "retinaresc"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type = str, help = "Path of the directory where the dataset is stored")
    parser.add_argument("--categories", type = str, nargs = "+", default = categories, help = "Dataset categories to perform compression on")
    parser.add_argument("--device", type = str, help = "Where to run the script")
    parser.add_argument("--backbone", type = str, help = "CNN model to use for feature extraction")
    parser.add_argument("--layer_idxs", type = str, nargs = "+", help = "List of layers to use for extraction")
    parser.add_argument("--compression_method", type = str, default = "JPEG", help = "Compression method to use on images, e.g. JPEG, PNG, ...")
    parser.add_argument("--quality", type = int, default = 50, help = "Amount of compression to be applied to the image when compressing with JPEG")
    parser.add_argument("--feature_dtype", type = str, default = "float32", help = "Data type of the features")
    parser.add_argument("--pq_method", type = str, default = "None", help = "Choose whether to perform product quantization globally or layer-wise. By default it doesn't apply compression")
    parser.add_argument("--use_layerwise_pq", action = "store_true", help = "If true, uses product quantization layer-wise rather than on the whole feature vector")
    parser.add_argument("--pq_subspaces", type = int, help = "Number of subspaces to use in product quantization")
    parser.add_argument("--centroids_per_subspace", type = int, default = 256, help = "Number of centroids per subspace to be used in PQ") 
    parser.add_argument("--seed", type = int, default = 1, help = "Execution seed")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed = args.seed
    device = torch.device(args.device)

    dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64}

    if args.feature_dtype not in dtype_mapping:
        raise ValueError(f"Unsupported feature_dtype: {args.feature_dtype}. Choose from {list(dtype_mapping.keys())}")

    args.feature_dtype = dtype_mapping[args.feature_dtype]

    compress_features(args.dataset_path, args.categories, device, args.backbone, args.layer_idxs, args.compression_method, 
                      args.quality, args.feature_dtype, args.pq_method, args.pq_subspaces, args.centroids_per_subspace)


if __name__ == "__main__":
    main()







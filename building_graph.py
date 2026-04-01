import os
import torch
import torchvision.transforms as T
import torchvision.models as models
from torch_geometric.data import Data
from PIL import Image

def get_feature_extractor(backbone_name: str, device, target_dim=256):
    
    if backbone_name == "efficientnet_b0": #The Backbone can be switched out to utilise other feature extraction backbones such as ResNet or MobileNet
        base = models.efficientnet_b0(weights='DEFAULT')
        old_conv = base.features[0][0]
        
        # Creates new conv layer with 4 input channels
        new_conv = torch.nn.Conv2d(4, old_conv.out_channels, 
                                   kernel_size=old_conv.kernel_size, 
                                   stride=old_conv.stride, 
                                   padding=old_conv.padding, 
                                   bias=False)
        
        with torch.no_grad():
            # Copy RGB weights to first 3 channels
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Uses 3:4 slice to maintain 4D shape (N, 1, H, W) to match the mean's shape (N, 1, H, W)
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            
        base.features[0][0] = new_conv
        extractor = torch.nn.Sequential(base.features, torch.nn.AdaptiveAvgPool2d((1, 1)))
        out_dim = 1280
    else:
        raise ValueError(f"Backbone {backbone_name} not configured for 4-channel input.")

    projection = torch.nn.Linear(out_dim, target_dim)
    extractor.eval().to(device)
    projection.to(device)
    return extractor, projection, target_dim

def ExtractImageFeatures(rgbPath, edgePath, model, projection, transform, device):
    rgb = Image.open(rgbPath).convert("RGB")
    edge = Image.open(edgePath).convert("L")

    rgb_t = transform(rgb)
    edge_t = transform(edge)

    # Combine into a 4-channel tensor (C=4, H=224, W=224)
    combined = torch.cat([rgb_t, edge_t], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(combined)
        features = torch.flatten(features, start_dim=1)
        features = projection(features)

    return features.squeeze(0)

def resolve_dual_paths(imageDirectory, base_name):
   #Finds both RGB and Edge-Segmented images
    clean_name = os.path.splitext(base_name)[0]
    rgb_dir = os.path.join(imageDirectory, "rgb")
    edge_dir = os.path.join(imageDirectory, "edges")

    def find_file(directory, name):
        if not os.path.exists(directory): return None
        for f in os.listdir(directory):
            if f.startswith(name):
                return os.path.join(directory, f)
        return None

    rgb_p = find_file(rgb_dir, clean_name)
    edge_p = find_file(edge_dir, clean_name)

    if not rgb_p or not edge_p:
        raise FileNotFoundError(f"Missing image pair for {clean_name} in {imageDirectory}")
    
    return rgb_p, edge_p

def BuildGlobalGraphFromCSV(csv_df, imageDirectory, labelMap, device):
    featureExtractor, projection, _ = get_feature_extractor("efficientnet_b0", device)
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    node_features = []
    edge_index = []
    edge_labels = []
    node_id_map = {}
    cache = {}

    def get_node_id(image_name):
        if image_name not in node_id_map:
            node_id_map[image_name] = len(node_features)
            if image_name not in cache:
                r_p, e_p = resolve_dual_paths(imageDirectory, image_name)
                cache[image_name] = ExtractImageFeatures(r_p, e_p, featureExtractor, projection, transform, device)
            node_features.append(cache[image_name])
        return node_id_map[image_name]

    for _, row in csv_df.iterrows():
        try:
            src = get_node_id(row["current_image"])
            dst = get_node_id(row["destination_image"])
            edge_index.append([src, dst])
            edge_labels.append(labelMap[row["direction"]])
        except FileNotFoundError:
            continue # Skip missing images 

    x = torch.stack(node_features)
    edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)

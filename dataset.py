import torch
from torch.utils.data import Dataset
import pandas as pd
from building_graph import BuildGlobalGraphFromCSV
import os
from PIL import Image

#Gives one (1) global graph for the GCN to learn from
class NavGraphDataset:
    # Ensures this matches the number of arguments passed (3 + self)
    def __init__(self, CSVpath, imageDirectory, device): 
        self.df = pd.read_csv(CSVpath)
        directions = sorted(self.df["direction"].unique())
        self.LabelMap = {d: i for i, d in enumerate(directions)}
        
        # Calls  building_graph.py logic (handles the 4-channel logic)
        self.graph = BuildGlobalGraphFromCSV(
            self.df, imageDirectory, self.LabelMap, device
        )

    def get_graph(self):
        return self.graph #Returns global graph to feed into GCN 
        #the GCN is only training ON edges NOT individual image
    
#Processed edge-segmented images
class ImageDataset(Dataset):
    #RGB + Edge-Segmented images
    def __init__(self, csv_file, rgb_dir, edge_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.rgb_dir = rgb_dir
        self.edge_dir = edge_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image"]

        #Pathbuilding for corresponding images
        rgb_path = os.path.join(self.rgb_dir, img_name)
        edge_path = os.path.join(self.edge_dir, img_name)

        rgb = Image.open(rgb_path).convert("RGB")
        edge = Image.open(edge_path).convert("L")  #Single channel

        if self.transform:
            rgb = self.transform(rgb)
            edge = self.transform(edge)

        #Concatenates the channels: [4, H, W], 4-channel tensor
        image = torch.cat([rgb, edge], dim=0)

        label = self.data.iloc[idx]["label"] #Target value
        return image, label

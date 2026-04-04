# GNN-Federated-Visual-Homing-for-Cooperative-Multi-Robot-Navigation-in-GPS-Denied-Environments

## Description: 
A fundamental challenge for cooperative multi-robot systems is reliable visual navigation in GPS-denied environments. This challenge is especially prevalent when robots operate under heterogenous sensing, computational, and environmental conditions. This model presents a privacy preserving framework for cooperative visual homing, combining decentralised learning with topology-aware scene understanding. The model is a hybrid Convolutional Neural Network (CNN) and Graph Neural Network (GNN), that represents the environment as a topological graph where image embeddings from nodes and navigational transitions form labeled edges for relational reasoning. The CNN portion of the model is frozen, the Graph Convolutional (GCN=GNN+CNN) learns from the graph that the frozen CNN feature extractor helps build.  

### SPIE Paper: [insert link]

### Thesis Paper: [insert link]

### Link to CNN Federated Visual Homing for Cooperative Multi-Robot Navigation in GPS Denied Environments by [@natitedros](https://github.com/natitedros): https://github.com/natitedros/WAVN-Federated-Learning

## Installation:

### Data Collection: 
This model's data was collected from a Gazebo environment using an automated script to collect images. The images were processed and paired using a Python script. 
1. The script for automating the collection of images by the nodes in Gazebo can be found under the file named **GazeboImageCollection.py**. The script was written by Dr. Damian Lyons (Fordham University). 
2. The script for processing and pairing the images, as well as generating the CSV file can be found in the file named **DataProcessing.py**. 

### Files: 
**building_graph.py:** This file includes the froxen feature extractor (EfficientNet B0) which can be changed out with any other CNN geature extractor (such as ResNet, MobileNet, etc). The image pairs, *current* and *destination*, are treated as connections in the global graph. This creates  a four-channel edge-augmented graph accepting RGB and Edge-Segmented images from the dataset (*images\images*). This file has four functions:

1. `get_feature_extractor` which defined the feature extraction backbone to be used, defined weights for the two image subsets, and then prepared the images to be extracted by defining the projection and target dimension.
2. `ExtractImageFeatures` transformed the RGB and Edge-Segmented images, and then combined the information into a 4-channel tensor.
3. `resolve_dual_paths` finds both the RGB and Edge-Segmented images and then ensures whether the image has a pair or not.
4. `BuildGlobalGraphFromCSV` creates the global graph with thenodes and edges based upon the three previous functions' information. 
  
**dataset.py:** This file reads the labels.csv file and loads the image pairs from *\images*. It also creates graph data samples using *building_graph.py*. This file has two functions:

  1. `NavGraphDataset` which gives one global graph for the hybrid model to learn from. As well as making sure the model only learns on the edges, not the individual images.
  2. `ImageDataset` processes the Edge-Segmented images and the RGB images to then pathbuild for the corresponding images. 
  
**GNNmodel.py:** This file defines the hybrid GNN architecture that is used to learn from the image pair graphs. It has one function:

1. `GNN` which is a 2-layer GCN that is followed by a global pooling and linear classification head. Global pooling aggregates the fetaures from all of the nodes in a graph into a single, fixed-size vector that represents the entire graph. Linear Classification ensures a single, connected layer that takes the final learned node or graph representations and maps them into the desired output classes. 
  
**train.py:** This file ensures that the GCN is 4-channel (accepting RGB and Edge-Segmented images). The model loads the dataset from labels.csv and then initailises the GNN model, following the initialisation it trains with x-entropy loss to evaluate the accuracy of the results/learning. This file has two main functions:

1. `train_with_cross_validation` which ensures the training has a 5-fold cross validation, stores metrics, and reports results.
2. `evaluate` ensures the standard evalusation for graph-based batches. 
  
**plot.py:** This is a simple file that provides a visual of the global graph that the GCN model learns from. 

## Usage:

### To Run:
  1. Ensure your dataset has been properly collected and labeled.
  2. Import your dataset to your IDE/Code environment.
  3. Check that variables match dataset labels.
  4. Setup vitural environment on terminal using *source/venv/bin*.
  5. Navigate to file using *cd/File_name*.
  6. Call *python train.py*.
  7. Results should show up in terminal. Visual representation of graph should appear in a separate window.

### Sample Run:
[Up(base) laptop ~ % source venv/bin/activate         
(venv) (base) laptop ~ % cd WAVN_Graph-Neural-Network-Code
(venv) (base) laptop ~ % WAVN_Graph-Neural-Network-Code % python train.py         

Fold 1/5
Epoch    0 | Loss 1.3881 | Acc: 25.15% | Rec: 25.62%
Epoch   50 | Loss 1.3516 | Acc: 32.10% | Rec: 32.18%
Epoch  100 | Loss 1.3260 | Acc: 36.47% | Rec: 36.50%
Epoch  150 | Loss 1.2810 | Acc: 39.61% | Rec: 39.62%
Epoch  200 | Loss 1.2208 | Acc: 43.42% | Rec: 43.35%
Epoch  250 | Loss 1.1854 | Acc: 45.58% | Rec: 45.67%
Epoch  300 | Loss 1.1063 | Acc: 51.70% | Rec: 51.75%
Epoch  350 | Loss 1.0600 | Acc: 52.98% | Rec: 53.00%
Epoch  400 | Loss 1.0075 | Acc: 54.37% | Rec: 54.37%
Epoch  450 | Loss 0.9871 | Acc: 57.20% | Rec: 57.23%
Epoch  500 | Loss 0.9599 | Acc: 56.48% | Rec: 56.44%
Epoch  550 | Loss 0.9102 | Acc: 58.80% | Rec: 58.80%
Epoch  600 | Loss 0.9114 | Acc: 58.59% | Rec: 58.64%
Epoch  650 | Loss 0.8888 | Acc: 60.44% | Rec: 60.44%
Epoch  700 | Loss 0.8772 | Acc: 60.34% | Rec: 60.37%
Epoch  750 | Loss 0.8486 | Acc: 61.78% | Rec: 61.83%
Epoch  800 | Loss 0.8367 | Acc: 62.04% | Rec: 62.07%
Epoch  850 | Loss 0.8253 | Acc: 63.27% | Rec: 63.34%
Epoch  900 | Loss 0.8159 | Acc: 64.20% | Rec: 64.19%
Epoch  950 | Loss 0.8013 | Acc: 63.43% | Rec: 63.47%
Best Metrics (Fold 1): Acc: 65.79%, Rec: 65.75%

Fold 2/5
Epoch    0 | Loss 1.3881 | Acc: 24.50% | Rec: 24.49%
Epoch   50 | Loss 1.3530 | Acc: 30.01% | Rec: 30.12%
Epoch  100 | Loss 1.3277 | Acc: 35.72% | Rec: 35.79%
Epoch  150 | Loss 1.2988 | Acc: 38.19% | Rec: 38.27%
Epoch  200 | Loss 1.2467 | Acc: 41.64% | Rec: 41.65%
Epoch  250 | Loss 1.1910 | Acc: 44.57% | Rec: 44.69%
Epoch  300 | Loss 1.1425 | Acc: 47.76% | Rec: 47.80%
Epoch  350 | Loss 1.1074 | Acc: 49.41% | Rec: 49.44%
Epoch  400 | Loss 1.0610 | Acc: 49.25% | Rec: 49.28%
Epoch  450 | Loss 1.0282 | Acc: 51.00% | Rec: 51.09%
Epoch  500 | Loss 0.9820 | Acc: 53.01% | Rec: 53.01%
Epoch  550 | Loss 0.9714 | Acc: 54.19% | Rec: 54.21%
Epoch  600 | Loss 0.9363 | Acc: 55.94% | Rec: 56.02%
Epoch  650 | Loss 0.9116 | Acc: 56.41% | Rec: 56.45%
Epoch  700 | Loss 0.9024 | Acc: 56.30% | Rec: 56.35%
Epoch  750 | Loss 0.8647 | Acc: 57.49% | Rec: 57.56%
Epoch  800 | Loss 0.8531 | Acc: 58.67% | Rec: 58.74%
Epoch  850 | Loss 0.8510 | Acc: 59.70% | Rec: 59.71%
Epoch  900 | Loss 0.8300 | Acc: 59.14% | Rec: 59.17%
Epoch  950 | Loss 0.8267 | Acc: 59.08% | Rec: 59.05%
Best Metrics (Fold 2): Acc: 62.12%, Rec: 62.23%

Fold 3/5
Epoch    0 | Loss 1.3891 | Acc: 24.96% | Rec: 25.08%
Epoch   50 | Loss 1.3474 | Acc: 31.65% | Rec: 31.66%
Epoch  100 | Loss 1.3229 | Acc: 34.84% | Rec: 34.81%
Epoch  150 | Loss 1.2885 | Acc: 36.90% | Rec: 36.90%
Epoch  200 | Loss 1.2251 | Acc: 41.95% | Rec: 41.96%
Epoch  250 | Loss 1.1596 | Acc: 44.11% | Rec: 44.10%
Epoch  300 | Loss 1.1037 | Acc: 47.56% | Rec: 47.56%
Epoch  350 | Loss 1.0485 | Acc: 50.33% | Rec: 50.34%
Epoch  400 | Loss 1.0247 | Acc: 51.93% | Rec: 51.93%
Epoch  450 | Loss 0.9830 | Acc: 53.99% | Rec: 53.97%
Epoch  500 | Loss 0.9437 | Acc: 54.71% | Rec: 54.71%
Epoch  550 | Loss 0.9148 | Acc: 57.28% | Rec: 57.27%
Epoch  600 | Loss 0.8857 | Acc: 57.64% | Rec: 57.64%
Epoch  650 | Loss 0.8784 | Acc: 60.22% | Rec: 60.21%
Epoch  700 | Loss 0.8596 | Acc: 58.93% | Rec: 58.93%
Epoch  750 | Loss 0.8354 | Acc: 60.58% | Rec: 60.59%
Epoch  800 | Loss 0.8270 | Acc: 60.58% | Rec: 60.57%
Epoch  850 | Loss 0.8223 | Acc: 60.37% | Rec: 60.37%
Epoch  900 | Loss 0.8128 | Acc: 61.25% | Rec: 61.25%
Epoch  950 | Loss 0.7837 | Acc: 61.91% | Rec: 61.92%
Best Metrics (Fold 3): Acc: 63.41%, Rec: 63.42%

Fold 4/5
Epoch    0 | Loss 1.3895 | Acc: 23.21% | Rec: 23.36%
Epoch   50 | Loss 1.3528 | Acc: 30.57% | Rec: 30.60%
Epoch  100 | Loss 1.3278 | Acc: 35.87% | Rec: 35.82%
Epoch  150 | Loss 1.2909 | Acc: 36.80% | Rec: 36.75%
Epoch  200 | Loss 1.2487 | Acc: 41.59% | Rec: 41.57%
Epoch  250 | Loss 1.1733 | Acc: 44.62% | Rec: 44.57%
Epoch  300 | Loss 1.1236 | Acc: 46.73% | Rec: 46.72%
Epoch  350 | Loss 1.0617 | Acc: 49.10% | Rec: 49.06%
Epoch  400 | Loss 1.0133 | Acc: 51.31% | Rec: 51.28%
Epoch  450 | Loss 0.9806 | Acc: 54.25% | Rec: 54.24%
Epoch  500 | Loss 0.9594 | Acc: 53.27% | Rec: 53.22%
Epoch  550 | Loss 0.9524 | Acc: 55.22% | Rec: 55.17%
Epoch  600 | Loss 0.9260 | Acc: 55.69% | Rec: 55.70%
Epoch  650 | Loss 0.8981 | Acc: 57.49% | Rec: 57.47%
Epoch  700 | Loss 0.8826 | Acc: 56.61% | Rec: 56.59%
Epoch  750 | Loss 0.8521 | Acc: 58.98% | Rec: 58.95%
Epoch  800 | Loss 0.8394 | Acc: 59.65% | Rec: 59.64%
Epoch  850 | Loss 0.8273 | Acc: 58.88% | Rec: 58.84%
Epoch  900 | Loss 0.8118 | Acc: 59.91% | Rec: 59.89%
Epoch  950 | Loss 0.8010 | Acc: 61.04% | Rec: 61.02%
Best Metrics (Fold 4): Acc: 62.58%, Rec: 62.56%

Fold 5/5
Epoch    0 | Loss 1.3887 | Acc: 26.30% | Rec: 26.00%
Epoch   50 | Loss 1.3489 | Acc: 31.24% | Rec: 31.60%
Epoch  100 | Loss 1.3200 | Acc: 35.51% | Rec: 35.65%
Epoch  150 | Loss 1.2866 | Acc: 36.49% | Rec: 36.69%
Epoch  200 | Loss 1.2311 | Acc: 39.94% | Rec: 40.01%
Epoch  250 | Loss 1.1616 | Acc: 45.29% | Rec: 45.45%
Epoch  300 | Loss 1.1158 | Acc: 47.09% | Rec: 47.17%
Epoch  350 | Loss 1.0656 | Acc: 49.46% | Rec: 49.39%
Epoch  400 | Loss 1.0056 | Acc: 51.16% | Rec: 51.08%
Epoch  450 | Loss 0.9756 | Acc: 53.73% | Rec: 53.67%
Epoch  500 | Loss 0.9496 | Acc: 55.84% | Rec: 55.77%
Epoch  550 | Loss 0.9341 | Acc: 56.00% | Rec: 56.02%
Epoch  600 | Loss 0.9182 | Acc: 54.50% | Rec: 54.42%
Epoch  650 | Loss 0.9024 | Acc: 56.46% | Rec: 56.47%
Epoch  700 | Loss 0.8695 | Acc: 56.97% | Rec: 56.94%
Epoch  750 | Loss 0.8621 | Acc: 58.05% | Rec: 57.96%
Epoch  800 | Loss 0.8453 | Acc: 58.57% | Rec: 58.59%
Epoch  850 | Loss 0.8459 | Acc: 58.47% | Rec: 58.44%
Epoch  900 | Loss 0.8290 | Acc: 59.39% | Rec: 59.45%
Epoch  950 | Loss 0.8150 | Acc: 59.60% | Rec: 59.57%
Best Metrics (Fold 5): Acc: 61.35%, Rec: 61.33%

--- Final Cross-Validation Results ---
Mean Accuracy:  63.05% ± 1.52
Mean Precision: 63.10% ± 1.50
Mean Recall:    63.06% ± 1.50
Mean F1-Score:  63.03% ± 1.51

Total runtime: 328.91 seconds
]()

### Example of graph visual: 
[WAVN_GCN_Graph.pdf](https://github.com/user-attachments/files/26418169/WAVN_GCN_Graph.pdf)


## Resources: 
1. Blumenkamp: (https://share.google/coTa8EHbpsbMQs4BR)
2. Paykari: (https://share.google/mwQolwX0GaA5rHAw6)
3. Tensor Flow: (https://www.tensorflow.org/federated)
4. PyG: (https://pytorch-geometric.readthedocs.io/en/latest/)

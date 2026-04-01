import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import time
from dataset import NavGraphDataset
from plot import plot_graph
from GNNmodel import GNN  
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

#Path configurations
CSVpath = "labels.csv"
imageDirectory = "images/images"  #Path to RGB and Edge-Segmented images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Experiment configuration...Not necessary since using 5-fold cross-validation. However, if removing k-fold cross validation, include learning configuration
experiment_configuration = {
    "epochs": 1000,
    "learningRate": 1e-3,
    "hiddenDimension": 96,
}

EPOCHS = experiment_configuration["epochs"]
LR = experiment_configuration["learningRate"]
hiddenDimension = experiment_configuration["hiddenDimension"]

def train_with_cross_validation(k_folds=5):
    dataset = NavGraphDataset(CSVpath, imageDirectory, device)
    data = dataset.get_graph().to(device)

    num_edges = data.edge_attr.size(0)
    edge_indices = torch.arange(num_edges)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    #Lists to store the best metrics from each fold (Printed with each run)
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(edge_indices)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        train_idx = torch.tensor(train_idx, device=device)
        val_idx = torch.tensor(val_idx, device=device)

        model = GNN(
            inChannels=data.x.shape[1],
            hiddenChannels=hiddenDimension,
            numClasses=len(dataset.LabelMap)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()
        
        #Initializes dictionary to track the best performing epoch for this fold
        best_val_metrics = {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            loss = criterion(out[train_idx], data.edge_attr[train_idx])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = out[val_idx].argmax(dim=1).cpu().numpy()
                val_true = data.edge_attr[val_idx].cpu().numpy()

                #Calculates the metrics for the current epoch
                acc = (val_preds == val_true).mean() * 100
                prec = precision_score(val_true, val_preds, average='macro', zero_division=0) * 100
                rec = recall_score(val_true, val_preds, average='macro', zero_division=0) * 100
                f1 = f1_score(val_true, val_preds, average='macro', zero_division=0) * 100

                #Updates the best metrics for the fold if accuracy improves
                if acc > best_val_metrics["acc"]:
                    best_val_metrics = {
                        "acc": acc, 
                        "prec": prec, 
                        "rec": rec, 
                        "f1": f1
                    }

            if epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | Loss {loss.item():.4f} | Acc: {acc:.2f}% | F1: {f1:.2f}%")

        #Reports the best results for this specific fold
        print(f"Fold {fold+1} Results -> Acc: {best_val_metrics['acc']:.2f}%, "
              f"Prec: {best_val_metrics['prec']:.2f}%, "
              f"Rec: {best_val_metrics['rec']:.2f}%, "
              f"F1: {best_val_metrics['f1']:.2f}%")
        
        fold_accuracies.append(best_val_metrics["acc"])
        fold_precisions.append(best_val_metrics["prec"])
        fold_recalls.append(best_val_metrics["rec"])
        fold_f1s.append(best_val_metrics["f1"])

    #Final summary of results
    print("\n" + "="*30)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("="*30)
    print(f"Mean Accuracy:  {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}")
    print(f"Mean Precision: {np.mean(fold_precisions):.2f}% ± {np.std(fold_precisions):.2f}")
    print(f"Mean Recall:    {np.mean(fold_recalls):.2f}% ± {np.std(fold_recalls):.2f}")
    print(f"Mean F1-Score:  {np.mean(fold_f1s):.2f}% ± {np.std(fold_f1s):.2f}")
    print("="*30)
    
    return data

def evaluate(model, loader):
    #Standard evaluation for graph-based batches
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
    return 100.0 * correct / total if total > 0 else 0

if __name__ == "__main__":
    start_time = time.perf_counter()
    data = train_with_cross_validation(k_folds=5)
    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")
    
    #Graph plotting, pops up in different window after experiment results have been shared
    try:
        plot_graph(data)
    except NameError:
        print("plot_graph not defined or plot.py missing.")

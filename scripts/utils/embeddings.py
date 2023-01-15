import os

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from umap import UMAP


def array_transforms(array):
    transforms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize([128, 128], interpolation=InterpolationMode.NEAREST),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
        ]
    )
    return transforms(array)


def extract_embeddings(model, data, save_dir: str, name: str):
    """Given a model and data, extract representations, projections, and predictions
    and plot UMAP embeddings of each. Save results to file.

    Parameters
    ----------
    model : _type_
        Trained SimSiam/FastSiam model (must have backbone, projection_head, and prediction_head attributes)
    data : DataFrame
        Dataframe containing waferMap and failureType/failureCode columns
    save_dir : str
        Root directory to save results to
    name : str
        Name of data for saved files, e.g. 'train_1' for 'train_1.csv',  by default ""
    """
    sns.set_theme()
    os.makedirs(save_dir, exist_ok=True)

    # Apply transforms to wafer map and stack tensors together
    inputs = data.waferMap.apply(array_transforms)
    inputs_tensor = torch.stack([_ for _ in inputs.values])

    reps = []
    projs = []
    preds = []

    # Frozen model: just use the model as an encoder
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(inputs_tensor.shape[0])):
            img_tensor = inputs_tensor[i]
            img_tensor = img_tensor.unsqueeze(dim=0)

            h = model.backbone(img_tensor).flatten(start_dim=1)
            z = model.projection_head(h)
            p = model.prediction_head(z)

            h = h.flatten()
            z = z.flatten()
            p = p.flatten()

            reps.append(h)
            projs.append(z)
            preds.append(p)

    # Cast tensor results to numpy arrays
    reps_stacked = torch.stack(reps).numpy()
    projs_stacked = torch.stack(projs).numpy()
    preds_stacked = torch.stack(preds).numpy()

    # Create dataframes for each representation/projection/prediction
    # Add failureType and failureCode columns as well
    df_reps = pd.DataFrame(reps_stacked)
    df_reps["failureType"] = data.failureType.values
    df_reps["failureCode"] = data.failureCode.values

    df_proj = pd.DataFrame(projs_stacked)
    df_proj["failureType"] = data.failureType.values
    df_proj["failureCode"] = data.failureCode.values

    df_pred = pd.DataFrame(preds_stacked)
    df_pred["failureType"] = data.failureType.values
    df_pred["failureCode"] = data.failureCode.values

    # Initialize titles for plots
    titles = ["Representations", "Projections", "Predictions"]

    raw_dfs = [df_reps, df_proj, df_pred]
    emb_dfs = []

    # Loop through representations/projections/predictions and plot UMAP embeddings
    for count, representations in enumerate(
        (reps_stacked, projs_stacked, preds_stacked)
    ):
        # UMAP embedding of scaled representations/projections/predictions
        reducer = UMAP(metric="cosine", random_state=42)
        scaler = StandardScaler()
        embedding = reducer.fit_transform(scaler.fit_transform(representations))
        emb_df = pd.DataFrame(
            {
                "umap_1": embedding[:, 0],
                "umap_2": embedding[:, 1],
                "failureType": data.failureType,
            }
        )
        # Sort by failureType to  get consistent colors in plots
        emb_df.sort_values(by="failureType", inplace=True)

        # Plot UMAP embedding
        ax = sns.scatterplot(
            data=emb_df, x="umap_1", y="umap_2", hue="failureType", alpha=0.3
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title(f"{name} {titles[count]}")
        plt.show()

        emb_dfs.append(emb_df)

    # Save representations/projections/predictions to file
    save_folders = ["reps", "proj", "pred"]
    for count, dfs in enumerate(zip(raw_dfs, emb_dfs)):
        df, emb_df = dfs
        os.makedirs(f"{save_dir}/{save_folders[count]}", exist_ok=True)
        df.to_csv(f"{save_dir}/{save_folders[count]}/{name}.csv", index=False)
        emb_df.to_csv(f"{save_dir}/{save_folders[count]}/{name}_umap.csv", index=False)

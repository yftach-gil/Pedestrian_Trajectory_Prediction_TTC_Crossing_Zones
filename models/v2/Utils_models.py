import torch
import numpy as np
import matplotlib.pyplot as plt

################## Metrics  ##################
def ade(pred, gt):
    return torch.mean(torch.norm(pred - gt, dim=1)) 
    # return torch.mean(torch.norm(pred - gt), dim=2)
def fde(pred, gt):
    # return torch.mean(torch.norm(pred[:, -1] - gt[:, -1], dim=1))
    return torch.mean(torch.norm(pred[:, -1] - gt[:, -1]))


################## Save and load torch model ##################
def load_model(file_name, model_class, device):
    """
    Load a model from checkpoint file.
    Switch to evaluation mode
    """
    checkpoint = torch.load(file_name, map_location=device)
    # Extract model constructor arguments from checkpoint
    model_kwargs = {k: v for k, v in checkpoint.items()
                    if k not in [
                        "model_state_dict", "model_name", "loss_history", "lr", "epochs", "batch_size", "penalty_spec"
                    ]}
    # Allow user to override any argument
    # model_kwargs.update(override_kwargs)
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    print(f"Loaded and switch to eval: {file_name}")
    return model, checkpoint

def get_model_hyperparameters(model):
    return {k: v for k, v in model.__dict__.items() if not k.startswith('_') and not callable(v)}

def save_model(model, model_name_str, file_name_to_save, loss_history, lr, epochs, batch_size, penalty_spec):
    """ Save a torch model checkpoint. including state_dict, hyperparameters, and training info."""
    hyperparams = get_model_hyperparameters(model)
    if 'training' in hyperparams:
        del hyperparams['training'] #not needed (status of model training\eval)
    checkpoint = {**hyperparams}
    # Add info from non-attributes of models (generally defined variables)
    checkpoint["model_name"] = model_name_str
    checkpoint["loss_history"] = loss_history
    checkpoint["lr"] = lr
    checkpoint["epochs"] = epochs
    checkpoint["batch_size"] = batch_size
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["penalty_spec"] = penalty_spec

    torch.save(checkpoint, file_name_to_save)
    print('file saved: ', file_name_to_save)
    return checkpoint


################## Evaluation ##################
def distribution_evaluation_for_extremes(ade_list, fde_list, filter_threshold_for_extremes):
    "from ade and fde list, analyze extreme values distributions."
    "Count how many values in list are greater than a given value."
    # filtered lists:
    ade_list_filtered = ade_list[ade_list > np.array(filter_threshold_for_extremes).flatten()]
    fde_list_filtered = fde_list[fde_list > np.array(filter_threshold_for_extremes).flatten()]
    #count greater values
    counts_greater_ade = np.sum(ade_list_filtered > filter_threshold_for_extremes)
    counts_greater_fde = np.sum(fde_list_filtered > filter_threshold_for_extremes)
    # print(f"Number of ADE values greater than {filter_threshold_for_extremes}: {counts_greater_ade}")
    # print(f"Number of FDE values greater than {filter_threshold_for_extremes}: {counts_greater_fde}")
    # statistics
    mean_ade_filtered = np.mean(ade_list_filtered)      # mean
    mean_fde_filtered = np.mean(fde_list_filtered)
    median_ade_filtered = np.median(ade_list_filtered)  # median
    median_fde_filtered = np.median(fde_list_filtered)
    max_ade_filtered = np.max(ade_list_filtered)        # max
    max_fde_filtered = np.max(fde_list_filtered)
    min_ade_filtered = np.min(ade_list_filtered)        # min
    min_fde_filtered = np.min(fde_list_filtered)
    # wrap
    distribution_evaluation_dict = {
        "ade_list_filtered":ade_list_filtered,
        "fde_list_filtered":fde_list_filtered,
        "counts_greater_ade": counts_greater_ade,
        "counts_greater_fde": counts_greater_fde,
        "mean_ade":   mean_ade_filtered,      "mean_fde":   mean_fde_filtered,
        "median_ade": median_ade_filtered,    "median_fde": median_fde_filtered,
        "max_ade":    max_ade_filtered,       "max_fde":    max_fde_filtered,
        "min_ade":    min_ade_filtered,       "min_fde":    min_fde_filtered
        }
    return distribution_evaluation_dict

def evaluate_model(model, dataloader, device, filter_threshold_for_extremes):
    """
    Calculates the Average ADE and FDE for all
    sequences in the batch, for all batches of dataloader.
    """
    model.eval()
    ade_list = []
    fde_list = []
    with torch.no_grad():
        # for batch_idx, (obs, fut) in enumerate(dataloader):
        for obs, fut in dataloader:    
            obs = obs.to(device)
            fut = fut.to(device)
            pred = model(obs)
            for i in range(pred.shape[0]):
                ade_val = ade(pred[i], fut[i])
                fde_val = fde(pred[i], fut[i])
                ade_list.append(ade_val)
                fde_list.append(fde_val)
    mean_ade = torch.mean(torch.tensor(ade_list))
    mean_fde = torch.mean(torch.tensor(fde_list))
    # Convert ade_list and fde_list from tensor to flat numpy array (or list)
    ade_list = np.array([x.item() if torch.is_tensor(x) else x for x in ade_list])
    fde_list = np.array([x.item() if torch.is_tensor(x) else x for x in fde_list])

    evaluation_dict = {"mean_ade":mean_ade,"mean_fde":mean_fde,"ade_list":ade_list,"fde_list": fde_list}

    # evaluation_dict_extremes is for filtered ade fde lists !!
    evaluation_dict_extremes=distribution_evaluation_for_extremes(
                                ade_list, fde_list, filter_threshold_for_extremes)

    return evaluation_dict, evaluation_dict_extremes 


################## Plot Loss ##################
def plot_loss_history(loss_history, EPOCHS, plot_title_detailes):
    """Plot the of in training
    given loss_history with three lists: [loss]
    and EPOCHS number
    """
    fig1 = plt.figure(figsize=(8, 4))
    # Plot Loss on left y-axis
    plt.scatter(range(1, EPOCHS+1), loss_history, label='Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value', color='black')
    plt.tick_params(axis='y', labelcolor='black')
    plt.legend()
    plt.title(f'Loss History {plot_title_detailes}')
    plt.grid()
    # plt.xticks(range(0, EPOCHS+1, 50))
    plt.show()

################## Plot samples from dataloader on graph ##################
def plot_model_predictions_sample_from_dataloader(model, dataloader, device, num_samples):
    """
    Plot samples from the dataloader on a simple xy graph.
    Args:
        model: The model to use for predictions.
        dataloader: The dataloader containing the data.
        device: The device to run the model on.
        num_samples: The number of samples to plot.
    """
    model.eval()
    with torch.no_grad():
        for obs, fut in dataloader:
            obs = obs.to(device)
            fut = fut.to(device)
            pred = model(obs).cpu()
            obs = obs.cpu()
            fut = fut.cpu()
            break # takes only first batch ?!

    ncols = 3
    nrows = int(np.ceil(num_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    point_size = 5
    line_width = 1

    for i in range(min(num_samples, obs.shape[0])):
        ax = axes[i]
        ax.plot(obs[i, :, 0],    obs[i, :, 1], color ='b',  linewidth=line_width,   label='History')
        ax.scatter(obs[i, :, 0], obs[i, :, 1], color='b',   s=point_size,           )
        
        ax.plot(fut[i, :, 0],    fut[i, :, 1], color='g',   linewidth=line_width,   label='Ground Truth Horizon')
        ax.scatter(fut[i, :, 0], fut[i, :, 1], color='g',   s=point_size,           )

        ax.plot(pred[i, :, 0],   pred[i, :, 1], color='r',  linewidth=line_width,   label='Predicted Horizon')
        ax.scatter(pred[i, :, 0], pred[i, :, 1], color='r', s=point_size,           )

        ade_sample = ade(pred[i], fut[i])
        fde_sample = fde(pred[i], fut[i])

        ax.set_title(f"Sample {i+1}\n obs_len={obs.shape[1]}, pred_len={fut.shape[1]}\n ADE={ade_sample:.2f}, FDE={fde_sample:.2f}")
        ax.axis('equal')
        ax.legend()
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


################## Plot samples from csv on aerial image ##################
## load background images
import os
import pandas as pd
def load_background_images_path(recordingId_list):
    """ load recording list and return images path list """
    image_directory = r'C:\Users\yftac\Documents\00_Project_Afeka\inD\drone-dataset-tools-master\data'
    image_paths = []
    for recording_id in recordingId_list:
        for file in os.listdir(image_directory):
            if file.endswith('.png'): # get list of png images with relevant id's
                if recording_id in file:
                    image_paths.append(os.path.join(image_directory, file))
    return image_paths # list of image paths
## load recording IDs
def load_recordingId_list(directory):
    file_name_list = os.listdir(directory)
    recordingId_list = file_name_list.copy()
    for i in range(len(recordingId_list)):
        recordingId_list[i]=recordingId_list[i].replace('_data_flattened.csv', '')
    return recordingId_list

def load_dfs(file_name_list, root_dir, folder_name):
    # os.chdir(root_dir)
    # change to data dir
    os.chdir(folder_name)
    dfs = []
    for i in range(len(file_name_list)):
        df=pd.read_csv(file_name_list[i])
        dfs.append(df)
    # change back to parent directory
    os.chdir(root_dir)
    return dfs

# verify that trackId_to_plot_list is valid
def check_tracks_and_frames(recordingId_to_plot, trackId_to_plot_list, frames_to_plot_first_index, dfs, recordingId_list, seq_len):
    """
    Checks if each trackId exists in the DataFrame for the given recordingId,
    if it belongs to class=='pedestrian', and if the requested frame range exists for each track.
    Returns a list of booleans for each trackId: True if valid, False otherwise.
    """
    print('If only this message is shown all is valid.')
    results = []
    if recordingId_to_plot not in recordingId_list:
        print(f"RecordingId {recordingId_to_plot} not found in recordingId_list.")
        return [False] * len(trackId_to_plot_list)
    recordingId_index = recordingId_list.index(recordingId_to_plot)
    df = dfs[recordingId_index]
    for i, trackId in enumerate(trackId_to_plot_list):
        track_df = df[df['trackId'] == trackId]
        if track_df.empty:
            print(f"TrackId {trackId} does not exist in recording {recordingId_to_plot}.")
            results.append(False)
            continue
        # Check if the track belongs to class 'pedestrian'
        if not (track_df['class'] == 'pedestrian').any():
            print(f"TrackId {trackId} in recording {recordingId_to_plot} is not a pedestrian.")
            results.append(False)
            continue
        first_frame_index = frames_to_plot_first_index[i]
        last_frame_index = first_frame_index + seq_len
        if last_frame_index > len(track_df):
            print(f"TrackId {trackId} in recording {recordingId_to_plot} does not have enough frames: requested {last_frame_index}, available {len(track_df)}.")
            results.append(False)
            continue
        results.append(True)
    return results
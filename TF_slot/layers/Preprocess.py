import torch.nn.functional as F


def rolling_average(data, window_size):
    """
    Compute the rolling average of the input data along the sequence length dimension.
    
    Parameters:
        data (torch.Tensor): Input tensor with shape (batch, seq_len, d_model)
        window_size (int): Size of the rolling window
    
    Returns:
        torch.Tensor: Output tensor with shortened seq_len due to rolling average
    """
    batch, seq_len, d_model = data.shape

    # Ensure the data is in the shape (batch, d_model, seq_len) for F.unfold
    data = data.permute(0, 2, 1)  # Shape: (batch, d_model, seq_len)
    
    # Apply padding if necessary
    padding = (window_size - 1) // 2
    data_padded = F.pad(data, (padding, padding), mode='replicate')
    
    # Use unfold to create rolling windows
    unfolded_data = data_padded.unfold(2, window_size, 1)  # Shape: (batch, d_model, new_seq_len, window_size)
    
    # Compute the rolling average
    rolling_avg = unfolded_data.mean(dim=-1)  # Shape: (batch, d_model, new_seq_len)
    
    # Permute back to the original shape
    rolling_avg = rolling_avg.permute(0, 2, 1)  # Shape: (Batch, batch, new_seq_len, d_model)
    
    return rolling_avg

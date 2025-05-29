# model.py

import torch
import torch.nn as nn
from torchsummary import summary

# Directly subclass nn.LSTM to control its forward pass for torchsummary
class LSTMForSummary(nn.LSTM):
    def forward(self, input, hx=None):
        # Call the original nn.LSTM forward method
        output, (h_n, c_n) = super().forward(input, hx)
        # For torchsummary, we only need to return the 'output' tensor.
        # This prevents the 'tuple' object has no attribute 'size' error.
        return output

class CNNLSTM(nn.Module):
    # Updated: Added num_classes as an argument
    def __init__(self, input_channels=2, timesteps=17, num_classes=3):
        super(CNNLSTM, self).__init__()
        
        self.timesteps = timesteps # Store timesteps for calculations
        self.input_channels = input_channels # Store input_channels for clarity/debugging
        self.num_classes = num_classes # Store num_classes
        
        # CNN layers (1D CNN for time series)
        # Input to Conv1d expects (batch_size, in_channels, seq_len)
        # Here, 'in_channels' corresponds to 'features_per_step' after permute
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2), # Output length: timesteps // 2

            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2), # Output length: (timesteps // 2) // 2 = timesteps // 4

            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2), # Output length: (timesteps // 4) // 2 = timesteps // 8
        )
        
        # Calculate the output sequence length after the CNN layers
        # For timesteps=17, after 3 MaxPool1d(2): 17 // 2 = 8, 8 // 2 = 4, 4 // 2 = 2
        cnn_output_timesteps = self.timesteps // (2**3) # Equivalent to self.timesteps // 8

        if cnn_output_timesteps < 1:
            raise ValueError(
                f"CNN output sequence length is too small ({cnn_output_timesteps}). "
                f"Original timesteps: {self.timesteps}. Please adjust timesteps or CNN MaxPool layers."
            )

        # Use our custom LSTM subclass for the summary tool
        # LSTM input_size is the out_channels of the last CNN layer (128)
        self.lstm = LSTMForSummary(input_size=128, hidden_size=64, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.num_classes) # Changed to use num_classes
        )
        
    def forward(self, x):
        # Input x shape: (batch_size, timesteps, features_per_step) e.g., (N, 17, 2)
        
        # Permute to (batch_size, features_per_step, timesteps) for Conv1d
        # This aligns the 'features_per_step' with 'in_channels' of Conv1d
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len) -> (N, 2, 17)
        
        cnn_out = self.cnn(x)  # Output will be (batch, 128, seq_len_after_cnn)
                                # For (N, 2, 17) input -> (N, 128, 2) output after CNNs
        
        # Permute back for LSTM input: (batch_size, seq_len, features)
        # Here, 'seq_len' is the reduced timesteps after CNN, and 'features' is 128
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, seq_len_after_cnn, 128) -> (N, 2, 128)
        
        # Pass through our custom LSTM which now returns only the output tensor
        # lstm_out shape: (batch, seq_len_after_cnn, hidden_size) -> (N, 2, 64)
        lstm_out = self.lstm(cnn_out) 
        
        # Take the output from the last timestep for the final classification
        # (batch, hidden_size) -> (N, 64)
        last_out = lstm_out[:, -1, :]  
        
        out = self.fc(last_out) # (batch, num_classes)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with the new parameters: input_channels=2, timesteps=17, num_classes=3
    # Assuming 3 stress levels: no stress, interruption, time pressure
    test_num_classes = 3 
    model = CNNLSTM(input_channels=2, timesteps=17, num_classes=test_num_classes).to(device)
    print(model)

    print("\nModel Summary:")
    # input_size should be (timesteps, input_channels) as per your forward method's initial permute
    summary(model, input_size=(17, 2)) # Adjusted input_size for summary

    # You can also test with a dummy input
    dummy_input = torch.randn(1, 17, 2).to(device) # Adjusted dummy input shape
    output = model(dummy_input)
    print(f"\nOutput shape from a dummy input: {output.shape}") # Should be torch.Size([1, test_num_classes])
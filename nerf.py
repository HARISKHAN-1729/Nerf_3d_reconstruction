
import torch
import torch.nn as nn

class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=256):
        super(NerfModel, self).__init__()
        
        # Define the first block of layers
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        
        # Define the second block of layers for density estimation
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)  # +1 for sigma
        )
        
        # Define the third block of layers for color estimation
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6, hidden_dim // 2), nn.ReLU()
        )
        
        # Define the fourth block of layers for final color output
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid()  # 3 for RGB channels
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        """
        Computes the positional encoding for input tensor x.
        
        Parameters:
        x (torch.Tensor): Input tensor.
        L (int): Number of frequency bands for positional encoding.
        
        Returns:
        torch.Tensor: Positional encoded tensor.
        """
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        """
        Forward pass through the NeRF model.
        
        Parameters:
        o (torch.Tensor): Input tensor for positions.
        d (torch.Tensor): Input tensor for directions.
        
        Returns:
        torch.Tensor, torch.Tensor: Predicted colors and densities.
        """
        # Compute positional encodings for the input positions and directions
        emb_x = self.positional_encoding(o, self.embedding_dim_pos) # emb_o: [batch_size, embedding_dim_pos * 6]
        emb_d = self.positional_encoding(d, self.embedding_dim_direction) # emb_d: [batch_size, embedding_dim_direction * 6]
        
        # Pass the positional encoding of positions through the first block of layers
        h = self.block1(emb_o) # h: [batch_size, hidden_dim]
        
        # Concatenate the output of block1 with the positional encoding and pass through block2
        tmp = self.block2(torch.cat((h, emb_o), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        
        # Split the output of block2 into features and sigma
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        
        # Concatenate the features with the positional encoding of directions and pass through block3
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        
        # Pass the output of block3 through block4 to get the final color
        c = self.block4(h) # c: [batch_size, 3]
        
        # Return the predicted color and density (sigma)
        return c, sigma

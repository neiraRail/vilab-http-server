import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo Autoencoder Convolucional simple
class CNN_Autoencoder(nn.Module):
    def __init__(self, latent_size):
        super(CNN_Autoencoder, self).__init__()
        self.latent_size = latent_size

        # Encoder: Convolutional layers to reduce the dimensionality
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 16, 100)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output shape: (batch_size, 16, 50)
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 32, 50)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output shape: (batch_size, 32, 25)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 64, 25)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # Output shape: (batch_size, 64, 12)
        )

        # Latent representation (fully connected layer to compress to a vector)
        self.fc1 = nn.Linear(64 * 12, 128)  # Flatten to a 1D vector (batch_size, 768)
        if self.latent_size < 128:
          self.fc2 = nn.Linear(128, latent_size)  # Latent space dimension
          self.fc3 = nn.Linear(latent_size, 128)  # Latent to 128
        else:
          self.fc2 = nn.Linear(128, 32)  # Latent space dimension
          self.fc3 = nn.Linear(32, 128)


        # Decoder: Transpose convolutions to reconstruct the signal
        
        self.fc4 = nn.Linear(128, 64 * 12)  # Reshape back to the feature map
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0),  # Output shape: (batch_size, 32, 25)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Output shape: (batch_size, 32, 50)

            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 16, 50)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Output shape: (batch_size, 16, 100)

            nn.ConvTranspose1d(in_channels=16, out_channels=6, kernel_size=3, stride=1, padding=1)  # Output shape: (batch_size, 6, 100)
        )

    def forward(self, x):
        # Transpose input from (batch, seq_len, features) to (batch, features, seq_len)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        
        # Encoding
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, 64 * 12)
        x = self.fc1(x)

        #if less than 128
        if self.latent_size < 128:
          x = self.fc2(x)
          x = self.fc3(x)
        
        # # Decoding
        x_reconstructed = self.fc4(x)
        x_reconstructed = x_reconstructed.view(x_reconstructed.size(0), 64, 12)  # Reshape back to (batch_size, 64, 12)
        x_reconstructed = self.decoder(x_reconstructed)

        # Transpose output back from (batch_size, features, seq_len) to (batch_size, seq_len, features)
        x_reconstructed = x_reconstructed.transpose(1, 2)  # (batch_size, seq_len, features)
        
        return x_reconstructed

    def encode(self, x):
        """Encodes the input signal to its latent vector without decoding."""
        # Transpose input from (batch, seq_len, features) to (batch, features, seq_len)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        
        # Encoding
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, 64 * 12)
        x = self.fc1(x)
        if self.latent_size < 128:
          x = self.fc2(x)

        return x  # This is the latent vector (batch_size, 32)
        
# Autoencoder Variational con tamaño de los vectores mu y logvar parametricos
class VAECNNAutoencoder(nn.Module):
    def __init__(self, latent_size):
        super(VAECNNAutoencoder, self).__init__()

        # Encoder: Convolutional layers to reduce the dimensionality
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 16, 100)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output shape: (batch_size, 16, 50)
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 32, 50)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output shape: (batch_size, 32, 25)
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 64, 25)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # Output shape: (batch_size, 64, 12)
        )

        # Latent representation (fully connected layer to compress to a vector)
        self.fc1 = nn.Linear(64 * 12, latent_size)  # Flatten to a 1D vector (batch_size, 768)

        self.fc2 = nn.Linear(64 * 12, latent_size)  # Latent space dimension
        self.fc3 = nn.Linear(latent_size, 64 * 12)  # Latent to 128
  


        # Decoder: Transpose convolutions to reconstruct the signal        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0),  # Output shape: (batch_size, 32, 25)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Output shape: (batch_size, 32, 50)

            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),  # Output shape: (batch_size, 16, 50)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Output shape: (batch_size, 16, 100)

            nn.ConvTranspose1d(in_channels=16, out_channels=6, kernel_size=3, stride=1, padding=1)  # Output shape: (batch_size, 6, 100)
        )
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    def representation(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        return self.bottleneck(x)[0]

    def forward(self, x):
        # Transpose input from (batch, seq_len, features) to (batch, features, seq_len)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        
        # Encoding
        x = self.encoder(x)
        h = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, 64 * 12)

        z, mu, logvar = self.bottleneck(h)
        
        # Decoding
        z = self.fc3(z)
        x_reconstructed = z.view(z.size(0), 64, 12)  # Reshape back to (batch_size, 64, 12)
        x_reconstructed = self.decoder(x_reconstructed)

        # Transpose output back from (batch_size, features, seq_len) to (batch_size, seq_len, features)
        x_reconstructed = x_reconstructed.transpose(1, 2)  # (batch_size, seq_len, features)
        
        return x_reconstructed


# Modelo Autoencoder LSTM con uso de vector latente de encoder como hidden state de decoder.
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_size, latent_size)  # Transform hidden state to latent size

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # hidden: [1, batch_size, hidden_size]
        latent = self.hidden_to_latent(hidden[-1])  # latent: [batch_size, latent_size]
        return latent
    
class LSTMDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(LSTMDecoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)  # Transform latent to hidden size
        self.lstm = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Final output layer
        self.output_size = output_size

    def forward(self, latent, sequence_length):
        hidden = self.latent_to_hidden(latent).unsqueeze(0)  # Reshape to [1, batch_size, hidden_size]
        cell = torch.zeros_like(hidden)  # Initialize cell state as zeros
        outputs = []
        input_seq = torch.zeros((latent.size(0), 1, self.output_size)).to(device)  # Starting input for the decoder (adjust as needed)

        for _ in range(sequence_length):
            output, (hidden, cell) = self.lstm(input_seq, (hidden, cell))
            input_seq = self.fc(output)  # Pass output through the fully connected layer
            outputs.append(input_seq)

        return torch.cat(outputs, dim=1)
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, sequence_length):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_size)
        self.decoder = LSTMDecoder(latent_size, hidden_size, input_size)
        self.sequence_length = sequence_length

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent, self.sequence_length)
        return output


# Clasificador simple
class Classifier(nn.Module):
  def __init__(self, embedding_size, num_classes):
    super().__init__()
    self.classifier = nn.Sequential(
        nn.Linear(embedding_size, num_classes),
        # nn.LogSoftmax(1)
    )


  def forward(self, x):
    # print(f"shape of hidden: {hidden.shape}")
    return self.classifier(x)# TODO test with multiple LSTM layers
  
# Autoencoder LSTM que usa el hidden del encoder repetido como input del decoder y los estados ocultos son inicializados en zeros.
class Encoder(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.LSTM(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          batch_first=True  # True = (batch_size, seq_len, n_features)
                            # False = (seq_len, batch_size, n_features)
                            #default = false
        )

    def forward(self, x):
        # print(f'ENCODER input dim: {x.shape}')
        # x = x.reshape((batch_size, self.seq_len, self.input_size))
        # print(f'ENCODER reshaped dim: {x.shape}')
        _, (hidden, cell) = self.rnn1(x)
        return hidden[-1]
class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_size, n_features, num_layers):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.n_features = n_features
        self.num_layers = num_layers

        self.rnn1 = nn.LSTM(
          input_size=embedding_size,
          hidden_size=n_features,
          num_layers=num_layers,
          batch_first=True
        )
        # Version 1, va directo a la dimensión de destino
        # self.output_layer = nn.Linear(self.embedding_size, n_features)

    def forward(self, hidden):
        # Repetir el vector oculto 100 veces
        inputs = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        cell = torch.zeros(self.num_layers, hidden.shape[0], self.n_features).to(device)
        new_hidden = torch.zeros(self.num_layers, hidden.shape[0], self.n_features).to(device)
        # inputs = torch.zeros(batch_size, self.seq_len, hidden.shape[2]).to(device)
        # outputs = torch.zeros(batch_size, self.seq_len, input_size).to(device)

        output, (_, _) = self.rnn1(inputs, (new_hidden, cell))
        return output
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_size, num_layers):
        super(RecurrentAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_size = embedding_size

        self.encoder = Encoder(seq_len, n_features, embedding_size, num_layers).to(device)
        self.decoder = Decoder(seq_len, embedding_size, n_features, num_layers).to(device)
    def forward(self, x):
        hidden = self.encoder(x)

        outputs = self.decoder(hidden)
        return outputs
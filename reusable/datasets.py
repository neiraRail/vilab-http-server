import torch
from torch.utils.data import Dataset

class EncodedTimeSeriesDataset(Dataset):
    def __init__(self, encoded_dataframe):
        """
        Args:
            encoded_dataframe (pd.DataFrame): DataFrame que contiene las versiones codificadas
                                               de las series de tiempo y los targets.
        """
        self.encoded_dataframe = encoded_dataframe.reset_index(drop=True)
        self.features = [col for col in encoded_dataframe.columns if col != 'target']

    def __len__(self):
        # El tamaño del dataset es el número de filas codificadas
        return len(self.encoded_dataframe)

    def __getitem__(self, idx):
        # Obtener los vectores codificados y la etiqueta correspondiente
        encoded_vector = torch.tensor(self.encoded_dataframe.iloc[idx][self.features].values, dtype=torch.float32)
        target = torch.tensor(self.encoded_dataframe.iloc[idx]['target'], dtype=torch.float32)

        return encoded_vector, target
    

class LabeledTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, window_size, return_file=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.window_size = window_size
        self.features = [col for col in dataframe.columns if col != 'target' and col != 'file']
        self.return_file = return_file

    def __len__(self):
        # The length of the dataset is the number of windows we can extract
        return len(self.dataframe) // self.window_size

    def __getitem__(self, idx):
        # Get a window of data starting from index `idx`
        idx = idx * self.window_size
        window = self.dataframe.iloc[idx:idx + self.window_size]
        # Extract the features and convert to a torch tensor
        features_tensor = torch.tensor(window[self.features].values, dtype=torch.float32)
        # Extract the target and convert to a torch tensor
        target_tensor = torch.tensor(window['target'].values[0], dtype=torch.float32)
        if self.return_file:
            file_tensor = torch.tensor(window['file'].values[0], dtype=torch.float32)
            return features_tensor, (target_tensor, file_tensor)

        return features_tensor, target_tensor
    
class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, window_size):
        self.dataframe = dataframe
        self.window_size = window_size
        self.features = [col for col in dataframe.columns if col != 'target' and col != 'file']

    def __len__(self):
        # The length of the dataset is the number of windows we can extract
        return len(self.dataframe) - self.window_size + 1

    def __getitem__(self, idx):
        # Get a window of data starting from index `idx`
        window = self.dataframe.iloc[idx:idx + self.window_size]
        # Extract the features and convert to a torch tensor
        features_tensor = torch.tensor(window[self.features].values, dtype=torch.float32)
        return features_tensor
        

# Custom Dataset class for prediction in 'future_steps' steps ahead.
class TimeSeriesPredictionDataset(Dataset):
    def __init__(self, dataframe, window_size, future_steps, history):
        self.dataframe = dataframe
        self.window_size = window_size
        self.features = [col for col in dataframe.columns if col != 'target']
        self.future_steps = future_steps
        self.history = history

    def __len__(self):
        # The length of the dataset is the number of windows we can extract
        # return len(self.dataframe) - self.window_size + 1
        return len(self.dataframe) - self.future_steps - self.history*self.window_size + 1

    def __getitem__(self, idx):
        # Get a window of data starting from index `idx`
        windows = []
        for i in range(self.history):
            window = self.dataframe.iloc[idx + i*self.window_size:idx + i*self.window_size + self.window_size]
            windows.append(torch.tensor(window[self.features].values, dtype=torch.float32))



        target_window = self.dataframe.iloc[idx + self.future_steps + (self.history-1)*self.window_size:idx + self.future_steps + self.history*self.window_size]

        # Extract the features and convert to a torch tensor
        features_tensor = torch.stack(windows)
        target = torch.tensor(target_window[self.features].values, dtype=torch.float32)
        return features_tensor, target
    

class EncodedTimeSeriesPredictionDataset(Dataset):
    def __init__(self, encoded_dataframe, future_steps, history):
        """
        Args:
            encoded_dataframe (pd.DataFrame): DataFrame que contiene las versiones codificadas
                                               de las series de tiempo y los targets.
        """
        self.encoded_dataframe = encoded_dataframe.reset_index(drop=True)
        self.features = [col for col in encoded_dataframe.columns if col != 'target']
        self.future_steps = future_steps
        self.history = history

    def __len__(self):
        # El tamaño del dataset es el número de filas codificadas
        return len(self.encoded_dataframe) - self.future_steps - self.history + 1

    def __getitem__(self, idx):
        # Obtener los vectores codificados y la etiqueta correspondiente
        encoded_vector = torch.tensor(self.encoded_dataframe.iloc[idx:idx+self.history][self.features].values, dtype=torch.float32)
        target         = torch.tensor(self.encoded_dataframe.iloc[idx + self.history + self.future_steps - 1][self.features].values, dtype=torch.float32)

        return encoded_vector, target
    
class AugmentedEncodedTimeSeriesPredictionDataset(Dataset):
    def __init__(self, Xs, ys):
        self.features = Xs
        self.targets = ys

    def __len__(self):
        # El tamaño del dataset es el número de filas codificadas
        return len(self.features)

    def __getitem__(self, idx):
        # Obtener los vectores codificados y la etiqueta correspondiente
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target         = torch.tensor(self.targets[idx], dtype=torch.float32)

        return features, target

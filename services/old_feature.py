# Este servicio esta pensado para permitir el procesamiento de la data y extraer características relevantes para las siguientes etapas de análisis.

# Tiene que haber metodos atomicos para cada característica que se quiera extraer.
# Tiene que haber un metodo que permita crear un vector de características a partir de un conjunto de métodos de extracción.
import pandas as pd
import numpy as np

def extract_features(data, feature_methods):
    """
    Extrae características de un conjunto de datos utilizando métodos específicos.

    Args:
        data (pd.DataFrame): DataFrame con los datos de entrada.
        feature_methods (list): Lista de funciones para extraer características.

    Returns:
        pd.DataFrame: DataFrame con las características extraídas.
    """
    features = {}
    
    for method in feature_methods:
        feature_name, feature_values = method(data)
        features[feature_name] = feature_values
    
    return pd.DataFrame(features)


# Características del dominio del tiempo
def time_domain_features(data):
    """
    Extrae características del dominio del tiempo.

    Args:
        data (pd.DataFrame): DataFrame con los datos de entrada.

    Returns:
        tuple: Nombre de la característica y sus valores.
    """
    features = {
        'mean': data.mean(),
        'std': data.std(),
        'max': data.max(),
        'min': data.min(),
        'median': data.median(),
        'var': data.var()
    }
    
    return 'time_domain', pd.Series(features)

# Características del dominio de la frecuencia
def frequency_domain_features(data):
    """
    Extrae características del dominio de la frecuencia.

    Args:
        data (pd.DataFrame): DataFrame con los datos de entrada.

    Returns:
        tuple: Nombre de la característica y sus valores.
    """
    fft_data = np.fft.fft(data, axis=0)
    features = {
        'fft_real': np.real(fft_data).mean(),
        'fft_imag': np.imag(fft_data).mean(),
        'fft_magnitude': np.abs(fft_data).mean(),
        'fft_phase': np.angle(fft_data).mean()
    }
    
    return 'frequency_domain', pd.Series(features)
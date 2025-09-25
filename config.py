# config.py
# Este módulo ahora carga la configuración desde config.yaml.

import yaml

def load_config():
    """Carga la configuración desde el archivo config.yaml."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: El archivo 'config.yaml' no fue encontrado.")
        return None
    except yaml.YAMLError as e:
        print(f"Error al parsear el archivo YAML: {e}")
        return None

# Carga la configuración una sola vez cuando se importa el módulo.
SETTINGS = load_config()
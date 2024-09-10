# name_generator.py
import random

def generate_names(brand_name, prefixes, suffixes, num_names=50):
    """
    Genera nombres de empresa combinando el nombre de marca o palabra favorita con prefijos y sufijos.
    """
    names = []
    for _ in range(num_names):
        prefix = random.choice(prefixes)  # Seleccionar un prefijo aleatorio
        suffix = random.choice(suffixes)  # Seleccionar un sufijo aleatorio
        # Combinar el prefijo, la palabra favorita del usuario, y el sufijo
        name = f"{prefix} {brand_name} {suffix}"
        names.append(name)
    
    return names

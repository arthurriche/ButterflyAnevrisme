def convert_to_float(data):
    """Convertit toutes les données d'un objet Data en float.

    Args:
        data (Data): L'objet Data à convertir.

    Returns:
        Data: L'objet Data avec toutes les données converties en float.
    """

    for key, value in data:
        if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
            data[key] = value.to(torch.float32)

    return data

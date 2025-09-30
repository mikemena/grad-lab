from logger import setup_logger

logger = setup_logger(__name__, include_location=True)

def flatten_dict(d, parent_key='', sep='_'):
    if not isinstance(d, dict):
        logger.warning(f"Expected a dictionary, got {type(d)}. Returning empty dict.")
        return {}

    items = []
    seen_keys = set()
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if new_key in seen_keys:
            logger.warning(f"Duplicate key detected: {new_key}")
        seen_keys.add(new_key)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def filter_numeric_metrics(metrics):
    flat_metrics = flatten_dict(metrics)
    return {k: v for k, v in flat_metrics.items() if isinstance(v, (int, float))}

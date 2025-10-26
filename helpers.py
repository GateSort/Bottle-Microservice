import json
from collections import defaultdict

def count_stickers_by_shape_and_color(sticker_results):
    """
    Count stickers by their shape and color combination.
    
    Args:
        sticker_results (list): List of dictionaries containing sticker information
                               Each dict should have 'Forma' (shape) and 'Color' keys
    
    Returns:
        dict: A dictionary with (shape, color) tuples as keys and counts as values
    """
    counts = defaultdict(int)
    
    for sticker in sticker_results:
        shape = sticker['Forma']
        color = sticker['Color']
        counts[(shape, color)] += 1
    
    return dict(counts)

def get_counts_as_json(sticker_results):
    """
    Get sticker counts in a JSON-friendly format.
    
    Args:
        sticker_results (list): List of dictionaries containing sticker information
    
    Returns:
        str: JSON string with counts organized by shape and color
    """
    counts = count_stickers_by_shape_and_color(sticker_results)
    
    # Reorganize the data for better JSON structure
    json_data = {
        "total": len(sticker_results),
        "counts": []
    }
    
    # Calculate counts by shape and color
    shape_counts = defaultdict(int)
    color_counts = defaultdict(int)
    
    for (shape, color), count in counts.items():
        shape_counts[shape] += count
        color_counts[color] += count
        json_data["counts"].append({
            "shape": shape,
            "color": color,
            "count": count
        })

    return json.dumps(json_data, indent=2)

from ultralytics import YOLO

def inspect_yolov8_layers(model_name="yolov8s.pt"):
    print(f"Inspecting {model_name}...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # YOLOv8 wraps the actual model in model.model
    # We want to inspect the Sequential list of layers
    layers = list(model.model.model.children())
    
    print("\nLayer Index | Type | Parameters | Source 'f'")
    print("-" * 50)
    for i, layer in enumerate(layers):
        layer_type = type(layer).__name__
        params = sum(p.numel() for p in layer.parameters())
        source = getattr(layer, 'f', 'N/A')
        
        # Highlight likely P3, P4, P5 candidates based on source and type
        highlight = ""
        if layer_type in ["C2f", "SPPF", "Conv"] and isinstance(source, int) and source == -1:
            if i in [4, 6, 9]:
                highlight = "<-- Common Hook Candidate"
                
        print(f"{i:11d} | {layer_type:15s} | {params:10,d} | {str(source):10s} {highlight}")

if __name__ == "__main__":
    inspect_yolov8_layers()

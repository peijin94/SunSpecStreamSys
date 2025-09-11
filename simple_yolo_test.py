#!/usr/bin/env python3
"""
Simple script to run YOLO prediction on latest_data.jpg
"""

import os
from ultralytics import YOLO

def main():
    # Paths
    model_path = 'model/best.pt'
    image_path = 'figs/latest_data.jpg'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    print(f"Running prediction on {image_path}")
    results = model.predict(image_path, conf=0.3, verbose=True)
    
    # Process results
    if results and len(results) > 0:
        result = results[0]
        print(f"\nFound {len(result.boxes) if result.boxes is not None else 0} detections")
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get bounding box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                
                # Convert to Python floats
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                
                # Get class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Map class ID to name
                class_name = 'type3' if class_id == 0 else 'type3b'
                
                print(f"Detection {i+1}: {class_name} (conf: {confidence:.3f}) at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        else:
            print("No detections found")
    else:
        print("No results returned")

if __name__ == "__main__":
    main()

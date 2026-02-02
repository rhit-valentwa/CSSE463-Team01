import numpy as np
from PIL import Image
from skimage import color
import os
import pathlib 

def analyze_resolution_variability(image_dir, num_samples=1000):
   
    print(f"Analyzing resolution variability from {num_samples} images...")
    
    image_files = list(pathlib.Path(image_dir).glob('*.jpg'))[:num_samples]
    
    widths = []
    heights = []
    aspect_ratios = []
    
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            width, height = img.size
            
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)
            
        except Exception as e:
            print(f"Error with {img_path}: {e}")
            continue
    
    print("RESOLUTION STATISTICS")
    print(f"Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}")
    print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}")
    print(f"Aspect Ratio - Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Mean: {np.mean(aspect_ratios):.2f}")
    
    landscape = sum(1 for ar in aspect_ratios if ar > 1.2)
    portrait = sum(1 for ar in aspect_ratios if ar < 0.8)
    square = len(aspect_ratios) - landscape - portrait
    
    print(f"Orientation distribution:")
    print(f"  Landscape: {landscape} ({100*landscape/len(aspect_ratios):.1f}%)")
    print(f"  Portrait:  {portrait} ({100*portrait/len(aspect_ratios):.1f}%)")
    print(f"  Square:    {square} ({100*square/len(aspect_ratios):.1f}%)")
    
    return widths, heights, aspect_ratios


def analyze_lab_distributions(image_dir, num_samples=500):

    print(f"Analyzing L*a*b* color distributions from {num_samples} images...")
    
    image_files = list(pathlib.Path(image_dir).glob('*.jpg'))[:num_samples]
    
    l_values = []
    a_values = []
    b_values = []
    
    for img_path in image_files:
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            
            lab_img = color.rgb2lab(img / 255.0)
            
            l_values.extend(lab_img[:, :, 0].flatten())
            a_values.extend(lab_img[:, :, 1].flatten())
            b_values.extend(lab_img[:, :, 2].flatten())
            
        except Exception as e:
            print(f"Error with {img_path}: {e}")
            continue
    
    # Convert to numpy arrays
    l_values = np.array(l_values)
    a_values = np.array(a_values)
    b_values = np.array(b_values)
    
    # Print statistics
    print("=== L*a*b* CHANNEL STATISTICS ===")
    print(f"L* (Lightness):")
    print(f"  Range: [{l_values.min():.1f}, {l_values.max():.1f}]")
    print(f"  Mean: {l_values.mean():.1f}, Std: {l_values.std():.1f}")
    
    print(f"a* (Green-Red):")
    print(f"  Range: [{a_values.min():.1f}, {a_values.max():.1f}]")
    print(f"  Mean: {a_values.mean():.1f}, Std: {a_values.std():.1f}")
    
    print(f"b* (Blue-Yellow):")
    print(f"  Range: [{b_values.min():.1f}, {b_values.max():.1f}]")
    print(f"  Mean: {b_values.mean():.1f}, Std: {b_values.std():.1f}")
    
    print("COLOR INSIGHTS")
    green_pixels = np.sum(a_values < -20)
    red_pixels = np.sum(a_values > 20)
    blue_pixels = np.sum(b_values < -20)
    yellow_pixels = np.sum(b_values > 20)
    
    total = len(a_values)
    print(f"Green-dominant pixels (a* < -20): {100*green_pixels/total:.1f}%")
    print(f"Red-dominant pixels (a* > 20): {100*red_pixels/total:.1f}%")
    print(f"Blue-dominant pixels (b* < -20): {100*blue_pixels/total:.1f}%")
    print(f"Yellow-dominant pixels (b* > 20): {100*yellow_pixels/total:.1f}%")
    
    return l_values, a_values, b_values


if __name__ == "__main__":
    path = "data/coco/train2017"  
    
    print("COCO 2017 Dataset Characteristics Analysis")
    print("=" * 50)
    
    # Check if path exists
    if not os.path.exists(path):
        print(f"Error: Dataset path not found: {path}")
        print("Please update path in the script to point to your COCO 2017 training images")
    else:
        widths, heights, aspect_ratios = analyze_resolution_variability(path, num_samples=1000)
        l_vals, a_vals, b_vals = analyze_lab_distributions(path, num_samples=500)
        
        print("\n" + "=" * 50)
        print("Analysis complete!")
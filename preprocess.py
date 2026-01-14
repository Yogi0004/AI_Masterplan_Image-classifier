import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def create_directory_structure(base_path):
    """Create train/val/test directory structure"""
    dirs = [
        'processed_data/train/masterplan',
        'processed_data/train/not_masterplan',
        'processed_data/val/masterplan',
        'processed_data/val/not_masterplan',
        'processed_data/test/masterplan',
        'processed_data/test/not_masterplan'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created: {full_path}")
    
    return os.path.join(base_path, 'processed_data')

def get_image_files(directory):
    """Get all image files from directory"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
    image_files = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(directory, file))
    
    return image_files

def balance_and_split_data(masterplan_path, non_masterplan_path, output_path):
    """Balance dataset and split into train/val/test"""
    
    # Get all images
    masterplan_images = get_image_files(masterplan_path)
    non_masterplan_images = get_image_files(non_masterplan_path)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Masterplan images: {len(masterplan_images)}")
    print(f"Non-Masterplan images: {len(non_masterplan_images)}")
    
    if len(masterplan_images) == 0 or len(non_masterplan_images) == 0:
        raise ValueError("One or both datasets are empty! Please check your paths.")
    
    # Balance the dataset - use minimum count
    min_count = min(len(masterplan_images), len(non_masterplan_images))
    
    # If datasets are very imbalanced, use at least 80% of smaller class
    target_count = min(min_count, max(len(masterplan_images), len(non_masterplan_images)))
    
    print(f"\n⚖️ Balancing dataset to {target_count} images per class")
    
    # Randomly sample to balance
    random.shuffle(masterplan_images)
    random.shuffle(non_masterplan_images)
    
    masterplan_images = masterplan_images[:target_count]
    non_masterplan_images = non_masterplan_images[:target_count]
    
    # Split data: 70% train, 15% val, 15% test
    mp_train, mp_temp = train_test_split(masterplan_images, test_size=0.3, random_state=42)
    mp_val, mp_test = train_test_split(mp_temp, test_size=0.5, random_state=42)
    
    nmp_train, nmp_temp = train_test_split(non_masterplan_images, test_size=0.3, random_state=42)
    nmp_val, nmp_test = train_test_split(nmp_temp, test_size=0.5, random_state=42)
    
    print(f"\n📦 Split Statistics:")
    print(f"Training: {len(mp_train)} masterplan, {len(nmp_train)} non-masterplan")
    print(f"Validation: {len(mp_val)} masterplan, {len(nmp_val)} non-masterplan")
    print(f"Testing: {len(mp_test)} masterplan, {len(nmp_test)} non-masterplan")
    
    # Copy files to new structure
    datasets = {
        'train': {'masterplan': mp_train, 'not_masterplan': nmp_train},
        'val': {'masterplan': mp_val, 'not_masterplan': nmp_val},
        'test': {'masterplan': mp_test, 'not_masterplan': nmp_test}
    }
    
    print("\n📁 Copying files...")
    for split, classes in datasets.items():
        for class_name, files in classes.items():
            dest_dir = os.path.join(output_path, split, class_name)
            for i, src_file in enumerate(files):
                ext = os.path.splitext(src_file)[1]
                dest_file = os.path.join(dest_dir, f"{class_name}_{i}{ext}")
                shutil.copy2(src_file, dest_file)
            print(f"  ✓ Copied {len(files)} files to {split}/{class_name}")
    
    print("\n✅ Dataset preparation complete!")
    return output_path

if __name__ == "__main__":
    # Define paths
    BASE_PATH = r"C:\Users\user\Desktop\MasterPlan_CLassification"
    MASTERPLAN_PATH = os.path.join(BASE_PATH, "Masterplan_Dataset")
    NON_MASTERPLAN_PATH = os.path.join(BASE_PATH, "Non_Masterplan_Dataset")
    
    print("🏗️ Masterplan Dataset Preprocessor")
    print("=" * 50)
    
    # Create directory structure
    output_path = create_directory_structure(BASE_PATH)
    
    # Balance and split data
    try:
        balance_and_split_data(MASTERPLAN_PATH, NON_MASTERPLAN_PATH, output_path)
        print(f"\n✨ Processed data saved to: {output_path}")
        print("\n🚀 You can now run train.py to train the model!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print(f"1. Masterplan path exists: {MASTERPLAN_PATH}")
        print(f"2. Non-Masterplan path exists: {NON_MASTERPLAN_PATH}")
        print("3. Both folders contain image files")
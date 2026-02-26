import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir='data/raw', dest_dir='data/processed', test_size=0.15, val_size=0.15):
    print("="*60)
    print("DATASET SPLITTING")
    print("="*60)
    
    for class_name in ['real', 'fake']:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Error: {class_path} not found!")
            continue
            
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Processing {class_name}: {len(images)} images found.")
        
        train_val, test = train_test_split(images, test_size=test_size, random_state=42)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
        
        for split, split_images in zip(['train', 'val', 'test'], [train, val, test]):
            target_path = os.path.join(dest_dir, split, class_name)
            os.makedirs(target_path, exist_ok=True)
            for img in split_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(target_path, img))
    
    print("\n✓ Dataset splitting complete!")

if __name__ == "__main__":
    split_dataset()
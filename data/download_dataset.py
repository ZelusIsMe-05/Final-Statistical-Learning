import os
import json
from datasets import load_dataset

def prepare_dataset_split(dataset_name, split_slice, folder_name, file_name):
    print(f"⏳ Loading dataset split: {split_slice}...")
    
    # Load the exact number of samples using slicing
    ds = load_dataset(dataset_name, split=split_slice)
    
    # Create the output directory if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Keep only essential fields to reduce file size
    # You can skip this step if you want to keep all columns (tags, link, etc.)
    data_list = [{"article": item["article"], "abstract": item["abstract"]} for item in ds]
    
    # Output file path
    output_path = os.path.join(folder_name, file_name)
    print(f"⏳ Saving {len(data_list)} samples to {output_path}...")
    
    # Save data to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Successfully saved: {output_path}\n")

if __name__ == "__main__":
    dataset_path = "nam194/vietnews"
    
    print("🚀 STARTING DATASET DOWNLOAD AND SPLIT...\n" + "="*50)
    
    # 1. Save 30,000 Train samples to the 'train' folder
    prepare_dataset_split(
        dataset_name=dataset_path, 
        split_slice="train[:30000]", 
        folder_name="train", 
        file_name="train_data.json"
    )
    
    # 2. Save 10,000 Validation samples to the 'validation' folder
    prepare_dataset_split(
        dataset_name=dataset_path, 
        split_slice="validation[:10000]", 
        folder_name="validation", 
        file_name="val_data.json"
    )
    
    # 3. Save 10,000 Test samples to the 'test' folder
    prepare_dataset_split(
        dataset_name=dataset_path, 
        split_slice="test[:10000]", 
        folder_name="test", 
        file_name="test_data.json"
    )
    
    print("🎉 PROCESS COMPLETED SUCCESSFULLY!")
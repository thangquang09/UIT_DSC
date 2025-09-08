from sklearn.model_selection import train_test_split
import pandas as pd
import os
from src.config import Config
from datasets import Dataset
from src.peft_model import get_trainer
from src.inference import generate


def main():
    df = pd.read_csv("data/vihallu-train.csv")

    train_df, val_df = train_test_split(df, test_size=Config.valid_size, random_state=Config.random_state, stratify=df[Config.label_column])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")    
        
    train_dataset =  Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    trainer = get_trainer(train_dataset, val_dataset)
    trainer.train()
    # Save the model after training
    trainer.save_model(Config.output_dir)
    print(f"Model saved to {Config.output_dir}")
    
    # Perform inference on train and test data
    print("Starting inference...")
    
    # Create output directories
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    
    # Load train data and perform inference
    print("Processing train data...")
    train_full_df = pd.read_csv("data/vihallu-train.csv")
    for idx, row in train_full_df.iterrows():
        sample = row.to_dict()
        output = generate(sample)
        
        # Save to file with id as filename
        with open(f"train/{row['id']}.txt", "w", encoding="utf-8") as f:
            f.write(output)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(train_full_df)} train samples")
    
    # Load test data and perform inference
    print("Processing test data...")
    test_df = pd.read_csv("data/vihallu-public-test.csv")
    for idx, row in test_df.iterrows():
        sample = row.to_dict()
        output = generate(sample)
        
        # Save to file with id as filename
        with open(f"test/{row['id']}.txt", "w", encoding="utf-8") as f:
            f.write(output)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} test samples")
    
    print("Inference completed!")
    print(f"Train outputs saved in 'train/' folder ({len(train_full_df)} files)")
    print(f"Test outputs saved in 'test/' folder ({len(test_df)} files)")


if __name__ == "__main__":
    main()

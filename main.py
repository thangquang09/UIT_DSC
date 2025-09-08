from sklearn.model_selection import train_test_split
import pandas as pd
from src.config import Config
from datasets import Dataset
from src.peft_model import get_trainer


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
    
    # sample_dict = df.iloc[0].to_dict()
    # prompt = create_finetuning_prompt(sample_dict)
    # print(prompt)
    # print(len(prompt.split()))
    # df['prompt'] = df.apply(create_finetuning_prompt, axis=1)
    # max_len = df['prompt'].apply(lambda x: len(x.split())).max()
    # min_len = df['prompt'].apply(lambda x: len(x.split())).min()
    # avg_len = df['prompt'].apply(lambda x: len(x.split())).mean()
    # print(f"Max length: {max_len}")
    # print(f"Min length: {min_len}")
    # print(f"Avg length: {avg_len}")


if __name__ == "__main__":
    main()

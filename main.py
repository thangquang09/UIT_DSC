from src.prompt import create_finetuning_prompt

import pandas as pd

def main():
    df = pd.read_csv("data/vihallu-train.csv")
    
    sample_dict = df.iloc[0].to_dict()
    prompt = create_finetuning_prompt(sample_dict)
    print(prompt)


if __name__ == "__main__":
    main()

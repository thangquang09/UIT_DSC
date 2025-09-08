import pandas as pd
from src.inference import generate

df = pd.read_csv("data/vihallu-train.csv")

sample = df.iloc[255]

label = sample['label']
sample['label'] = pd.NA
sample = sample.to_dict()
print(sample)
print("Label:", label)

ans = generate(sample=sample)

print(ans)
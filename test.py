# from src.inference import generate
import pandas as pd
from src.prompt import create_inference_prompt

df = pd.read_csv("data/vihallu-train.csv")

sample = df.iloc[0].to_dict()

print(create_inference_prompt(sample))
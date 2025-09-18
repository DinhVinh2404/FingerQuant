from train.train_model1 import ThumbDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd

def build_thumb_data(batch_size=32, worker=1, data_path=''):
    df = pd.read_pickle(data_path)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=['Handedness'], random_state=42)

    train_dataset = ThumbDataset(train_df)
    test_dataset = ThumbDataset(test_df)

    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle= True, num_workers=worker)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True, num_workers=worker)

    return train_loader, test_loader

import torch
import os
import pandas as pd
from torch.utils.data import Dataset

def get_year(series):
    nan_index = series.index[series.notna()].tolist()
    year = sorted(list(set([int(_[3]) for _ in nan_index])))
    year_col = ['L2Y'+str(y) for y in year]
    
    return year, year_col
    

class KELS(Dataset):
    def __init__(self, root_dir='./preprocessed/merge/outer'):
        self.root_dir = root_dir
        self.input_df = pd.read_csv(os.path.join(self.root_dir, 'input_merge.csv')).set_index('L2SID')
        self.label_df = pd.read_csv(os.path.join(self.root_dir, 'label_merge.csv')).set_index('L2SID')
   
    def __len__(self):
        if len(self.label_df) == len(self.input_df):
            return len(self.label_df)
    
        return False
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        year, year_col = get_year(self.label_df.iloc[idx])
        
        
        
        
        
        
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
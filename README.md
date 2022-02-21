# KELS

## Dataset
- dataset from 한국교육개발원 한국교육종단연구(KELS2013)
- https://www.kedi.re.kr/khome/main/research/requestResearchData.do

## Preprocessing
- merge questions in same category (with codebook)
- used only categorical data (not continuous data)
- fill missing value with average of column
- label for L2Y6: 1,2,3,4,5,6,7,8,9

## Dataloader
- dataset: KELS, sorted by L2SID
- sample {'year': [ year ], 'input':{columns_name:input}, 'label':{columns_name:label}}

## Baselines (ML algortimhs)
- Decision Tree
- Support Vector Machine
- Extra Trees
- Random Forest
- K-nearest neighbor
- Gradient Boosting

## Models
- MLP (2 Layer)
- LSTM
- Transformer (ViT)

## Code guide
- Download source code from git
```
git clone https://github.com/ujos89/KELS.git
cd KELS
```
- Add dataset
./dataset/KELS2013_ ... (.sav files)

- Data preprocessing & run code
```
python preprocessing.py
python ml.py
python mlp.py
```
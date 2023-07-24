from process_data import ProcessData
from clean_data import CleanData

train_df = CleanData('data/Train_Tagged_Titles.tsv').clean_data()

process_data = ProcessData(train_df)

#print first 50 items of id2tag
print({k: process_data.id2tag[k] for k in list(process_data.id2tag)[:50]})
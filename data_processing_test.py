from process_data import ProcessData
from clean_data import CleanData
from transformers import BertTokenizer

test_df = CleanData('data/Train_Tagged_Titles.tsv').test()

print(test_df.head())

def convert_tokens_tags_to_ids(df):
  new_df = df.copy()
  tz = BertTokenizer.from_pretrained('bert-base-cased')


  new_df['Title'] = new_df['Title'].apply(lambda x: tz.tokenize(x))
  print(new_df['Title'].head(50))
  new_df['Title'] = new_df['Title'].apply(lambda x : tz.convert_tokens_to_ids(x))

  return new_df

example = test_df['Title'].tolist()[0]
print(example)


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
test = tokenizer(example, padding='max_length', truncation=True, max_length=512, return_tensors='tf')
print(tokenizer.decode(test['input_ids'][0]))
print(tokenizer.convert_ids_to_tokens(test['input_ids'][0]))

#align label to pad deconstructed tokens

#align labeled tokens to deconstructed tokens referencing id of deconstructed tokens

#train on deconstructed tokens and labels
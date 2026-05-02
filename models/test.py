import os

from transformers import pipeline
from models import Sentiment_Model, News_Dataset
from sklearn.preprocessing import StandardScaler

ins = ['Привет', 'лох']

sentiment_config = {
            'model_name' : "blanchefort/rubert-base-cased-sentiment".strip(),
            'num_labels' : 3,
            'device' : 'cpu',
            'output_hidden_states' : True
            }

mdl_nm = 'seara/rubert-tiny2-russian-sentiment'
mdl = pipeline("text-classification",
               model=mdl_nm,
               tokenizer=mdl_nm)
out = mdl(ins)

print(out)


sentiment_config = {
            'model_name' : "blanchefort/rubert-base-cased-sentiment".strip(),
            'num_labels' : 3,
            'device' : 'cpu',
            'output_hidden_states' : True
            }

path_news_data = os.path.join('..', 'data', 'news_data')
news_data = News_Dataset(data_dir=path_news_data,
                             load_from_file=False,
                             delete_old=True)

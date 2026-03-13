import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


from itertools import islice
from corus import load_lenta, load_lenta2, load_mokoron, load_buriy_news, load_buriy_webhose
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from pathlib import Path
from typing import Callable, Generator, Any, Iterable
from abc import abstractmethod, ABCMeta

#global variable to map preprocessing functions to the files
func_to_data = {load_lenta: ['lenta-ru-news.csv.gz'],
                load_lenta2 : ['lenta-ru-news.csv.bz2'],
                load_mokoron : ['db.sql'],
                load_buriy_news : ['news-articles-2014.tar.bz2',
                                   'news-articles-2015-part1.tar.bz2',
                                   'news-articles-2015-part2.tar.bz2'],
                load_buriy_webhose : ['webhose-2016.tar.bz2'],
                pd.read_csv : [file for file in os.listdir(DATA_DIR) if file.endswith('.csv')]
                }


class Abstract_Fin_Dataset(Dataset, ABCMeta):

    def __init__(self,
                 data_dir: str | Path = os.path.join('..', 'data'),
                 batch_size: int = 32,
                 unified_file_nm: str = 'all_data.csv',
                 load_from_file: bool = False,
                 delete_old: bool = True,
                 ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_elements = None
        self.unified_file_nm = unified_file_nm
        self.delete_old = delete_old
        
        if not load_from_file:
            self.load()

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        '''
        Abstract function to load the dataset into a file

        Inputs:
            -

        Outputs:
            -
        '''
    
        ...

    def __len__(self) -> int:
        return int(np.ceil(self.num_elements / self.batch_size)) 

    def __iter__(self) -> list[int]:
        for idx in range(0, self.num_elements, self.batch_size):
            yield list(range(idx, min(idx+self.batch_size, self.num_elements)))


class News_Dataset(Abstract_Fin_Dataset):

    def __init__(self,
                 data_dir: str | Path = os.path.join('.'),
                 slice_size: int = 10_000,
                 unified_file_nm: str = 'all_data.csv',
                 load_from_file: bool = False,
                 delete_old: bool = True,
                 **kwargs
                 ) -> None:

        super().__init__()

        self.slice_size = slice_size
        self.unified_file_nm = unified_file_nm
        self.delete_old = delete_old

        self.load()


    def _prepare_data(self,
                      data_path: str | Path,
                      load_func: Callable,
                      verbose: bool = True
                      ) -> Generator[pd.DataFrame | None]:

        '''
        Function to prepare data of a single file (archive of csv) into a pd.DataFrame object

        Inputs:
            data_path: str | Path - path to the file
            load_func: Callable - function to load it with
            slice_size: int = 10_000 - size of a chunk to process at a time
            verbose: bool = True - verbosity flag

        Outputs:
            pd.DataFrame - chunk of the file

        '''

        end = data_path.split('.')[-1]

        match end:

            case 'csv':

                reader = load_func(data_path, chunksize=self.slice_size)
                for df in tqdm(reader,
                               desc='Loading DataFrame chunks',
                               leave=False,
                               disable=not verbose,
                               unit=' chunks'):
                    yield df


            case 'gz' | 'bz2' | 'sql':

                #make the object an iterable for the `islice` function to work
                gen = iter(load_func(data_path))
                
                lines_cnt = 0
                with tqdm(unit=' lines', leave=False, disable=not verbose) as progress:
                    while True: #because we don't know the num of elements in the generator
                        data = list(islice(gen, self.slice_size))
                        
                        if data is None or len(data) == 0:
                            break

                        columns = [col for col in data[-1].__attributes__
                                            if col in ['date', 'text', 'timestamp']
                                   ]
                    
                        df = pd.DataFrame(data, columns=data[-1].__attributes__)[columns]
                        yield df 
                        lines_cnt += df.shape[0]
                        self.num_elements += df.shape[0]
                        progress.update(self.slice_size) 

            case _:
                print(f'Unknown datatype to process: {data_path.split(".")}')
                yield None


    def load(self,
             verbose: bool = True
             ) -> None:

        '''
        Opens archives and saves their contents in a single file

        Inputs:
            self.data_dir:str | Path = DATA_DIR - inputs in the format (function_to_process_file, file_names)
            self.slice_size: int = 10_000 - size of a chunk DataFrame to load from each archive
            self.delete_old: bool = True - flag to delete the old file
            self.unified_file_nm: str = 'all_data.csv' - name of the new unified file
            verbose: bool = True - verbosity flag

        Outputs:
            None

        '''

        path_save = os.path.join(self.data_dir, self.unified_file_nm)

        if os.path.exists(path_save) and self.delete_old:
            if verbose:
                print('Found file with the same name in the data directory, deleting...')
            os.remove(path_save)
        

        header = True
        df_tmp = pd.DataFrame(columns=['date', 'text']).to_csv(path_save, header=header)
        header = False

        for load_func, data_paths in tqdm(func_to_data.items(),
                                          desc='Loading data',
                                          unit=' file batches',
                                          colour='green',
                                          disable=not verbose):

            for idx, filenm in tqdm(enumerate(data_paths),
                                    desc='Adding paths',
                                    leave=False,
                                    disable=not verbose):
                data_paths[idx] = os.path.join(self.data_dir, filenm) #integrate full path with filename
            
            data_path = data_paths[0]

            for data_path in tqdm(data_paths,
                                  desc=f'Loading archive {data_path}',
                                  leave=False,
                                  unit=" files",
                                  disable=not verbose):

                for df in self._prepare_data(data_path, load_func, self.slice_size, verbose):

                    if 'date' not in df.columns:
                        df['date'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.date())
                        
                    df.to_csv(path_save,
                              header=header,
                              mode='a',
                              columns=['date', 'text']
                              )

        return None 


    def __getitem__(self, idxs: int | list[int]) -> tuple[Any]:
        return tuple(pd.read_csv(self.unified_file_nm, skip_rows=idx, nrows=1).iloc[0, :])



#TODO: refactor the slop below into something that actually works

class Abstract_Sentiment_Model(nn.Module, ABCMeta):

    def __init__(self, model_name: str, *args, **kwargs) -> None:

        super().__init__()

        self.model_name = model_name


    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        ...

class Russian_Sentiment_Model(Abstract_Sentiment_Model):
    
    def __init__(self,
                 model_name: str = '',
                 num_labels: int= 3
                 ) -> None:

        super().__init__()
        
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
       
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self,
                input_ids: Iterable[int],
                attention_mask,
                labels=None
                ):
        
        '''
        Forward pass. If labels are provided, returns loss and logits.
        Otherwise returns only logits.
        '''

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        if labels is not None:
            return outputs.loss, outputs.logits
        else:
            return outputs.logits

    def predict(self, texts: Iterable[str], device: str = 'cpu'):
        '''
        Run inference on a list of texts.
        Returns predicted class indices and probabilities.
        '''

        self.eval()
        with torch.no_grad():
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            logits = self.forward(encodings['input_ids'], encodings['attention_mask'])
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds.cpu().numpy(), probs.cpu().numpy()


def train_model(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                loss, logits = model(input_ids, attention_mask, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return model

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description='Russian Sentiment Analysis with Transformers')
    parser.add_argument('--train_file', type=str, help='Path to training CSV (columns: text, label)')
    parser.add_argument('--eval_file', type=str, help='Path to evaluation CSV (optional)')
    parser.add_argument('--model_name', type=str, default='DeepPavlov/rubert-base-cased',
                        help='Pre-trained model name from Hugging Face')
    parser.add_argument('--num_labels', type=int, default=3, help='Number of sentiment classes')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max token length')
    parser.add_argument('--save_dir', type=str, default='./saved_model', help='Directory to save model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--predict', type=str, help='Text to predict sentiment (if given, runs inference)')
    args = parser.parse_args()

    # -------------------- Inference only mode --------------------
    if args.predict:
        # Load model (if saved, otherwise use pre-trained)
        model = RussianSentimentModel(args.model_name, args.num_labels)
        # If a saved model exists, load its weights
        if os.path.exists(args.save_dir):
            model.load_state_dict(torch.load(os.path.join(args.save_dir, 'pytorch_model.bin'), map_location='cpu'))
        model.to(args.device)
        preds, probs = model.predict([args.predict], device=args.device)
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}  # adjust if needed
        print(f'Text: {args.predict}')
        print(f'Predicted sentiment: {label_map[preds[0]]} (confidence: {probs[0][preds[0]]:.4f})')
        return

    # -------------------- Training mode --------------------
    if not args.train_file:
        print("Error: --train_file is required for training.")
        return

    # Load data
    df = pd.read_csv(args.train_file)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Split into train/val (80/20)
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, df['label'].values, test_size=0.2, random_state=42
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets and loaders
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = RussianSentimentModel(args.model_name, args.num_labels)

    # Train
    trained_model = train_model(model, train_loader, val_loader, args.device,
                                epochs=args.epochs, lr=args.lr)

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(args.save_dir, 'pytorch_model.bin'))
    # Also save tokenizer for later use
    tokenizer.save_pretrained(args.save_dir)
    print(f"Model saved to {args.save_dir}")

    # Optional evaluation on separate test file
    if args.eval_file:
        df_test = pd.read_csv(args.eval_file)
        test_dataset = SentimentDataset(df_test['text'].values, df_test['label'].values, tokenizer, args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        trained_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device)
                logits = trained_model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print("\nTest Set Results:")
        print(classification_report(all_labels, all_preds, target_names=['negative', 'neutral', 'positive']))

if __name__ == '__main__':
    main()



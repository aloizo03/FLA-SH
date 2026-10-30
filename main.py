import sys
import math

import numpy as np
import os
import cv2
import re

from datetime import datetime
from collections import Counter
import random
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tempfile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import HashingVectorizer
from PIL import Image

import pandas as pd

from tqdm import tqdm

from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import CLIPProcessor, CLIPModel

import torch
import time

from transformers import pipeline
from transformers import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.set_verbosity_warning()

import itertools
import glob
import _pickle as pkl

from utils_dataShapley import create_model, scale_to_grid, merge_small_chunks, _shapley_log_p, _chunk_inclusion_probs, _log_bernoulli_q, _sample_bernoulli_coalition, _sample_uniform_size_coalition, _sample_coalition_qtilde
from utils import get_metric, transform_to_torch, drop_index_columns, FeatureTransform, PREPROCESSING_MODES




# Use cache to not reapet the expensive embedding computations for the same image or text file across multiple runs of the script. 
# The cache is keyed by absolute file path, so if the same file is used in multiple datasets, it will only be embedded once per run.
_IMG_BASE_EMBED_CACHE = {}
_IMG_FAIR_TXT_EMBED_CACHE = {}
_CLIP_MODEL_CACHE = {}
_IMG_PIXEL_CACHE = {}



class Data_Shapley_Dataset:

    def __init__(self, 
                 datasets_path, 
                 outpath, 
                 metric='Accuracy', 
                 problem='classification', 
                 model_type='MLP_Class', 
                 C=1, 
                 hidden_layers=[32, 32], 
                 num_of_classes=None,
                 use_fairness=False,
                 label_class='class',
                 fairness_attr=0,
                 num_of_groups=2, 
                 lambda_fair=1.0, 
                 use_embeds=False, 
                 fair_metric='groupdro',
                 preprocessing='onehot_zscore',
                 utility='accuracy'):
        
        self.datasets_path = datasets_path
        self.metric = metric
        self.problem = problem
        self.model_type = model_type
        self.C = C
        self.outpath = outpath
        self.num_of_classes = num_of_classes
        self.hidden_layers = hidden_layers

        self.model = create_model(self.model_type, problem=self.problem, max_epochs=1, batch_size=1,  lr=0.0001, val_frac=0.1, hidden_layer_size=self.hidden_layers, num_classes=self.num_of_classes)
        self.values_ds_FLA_SH_run = None
        self.last_fla_sh_rounds_run = None
        self.mc_ds = None
        self.vanilia_ds = None
        self.datasets_filenames = {}
        self.label_to_idx = {}
        self.use_img = False
        self.use_embeds = use_embeds
        
        self.label_class = label_class
        self.use_fairness = use_fairness
        self.fair_metric = fair_metric

        self.fairness_attr = fairness_attr
        self.num_of_groups = num_of_groups

        self.lambda_fair = lambda_fair

        assert preprocessing in PREPROCESSING_MODES, preprocessing
        self.preprocessing = preprocessing
        self.feature_transform = None
        assert utility in ('accuracy', 'neg_val_ce'), utility
        self.utility = utility

    def read_data(self, class_label="class"):
        
        if os.path.exists(self.datasets_path):
            train_files = glob.glob(self.datasets_path + "/*.csv")
            datasets_train_df = {file_.split('/')[-1].split('.')[0].upper(): {'X': None, 'Y': None} for file_ in
                                 train_files}
            self.datasets_filenames = {file_.split('/')[-1].split('.')[0].upper(): file_ for file_ in train_files}
            for file_ in train_files:
                df = drop_index_columns(pd.read_csv(file_, index_col=False))
                file_name = file_.split('/')[-1].split('.')[0].upper()
                df = df.fillna(-99)
                class_labels = df.loc[:, df.columns.str.startswith(class_label)]
                row_tuples = df.drop(columns=list(class_labels.columns))
                datasets_train_df[file_name]['X'] = row_tuples
                datasets_train_df[file_name]['Y'] = class_labels
            return datasets_train_df
        else:
            AssertionError('Error')

    def read_img_data(self, class_label="class"):
        if os.path.exists(self.datasets_path):
            train_files = glob.glob(self.datasets_path + "/*.csv")
            datasets_train_dict = {file_.split('/')[-1].split('.')[0].upper(): {'X': None, 'Y': None} for file_ in
                                 train_files}
            self.datasets_filenames = {file_.split('/')[-1].split('.')[0].upper(): file_ for file_ in train_files}

            for file_ in train_files:
                df = pd.read_csv(file_, index_col=False)
                file_name = file_.split('/')[-1].split('.')[0].upper()
                class_labels = df.loc[:, df.columns.str.startswith(class_label)]
                X_lst = []
                target_col = None
                for col in ['file_path', 'file', 'filename']:
                    if col in df.columns:
                        target_col = col
                        break
                        
                if target_col is None:
                    raise KeyError("Dataframe must contain a column named 'file_path', 'file', or 'filename'")
                
                for idx, row in df.iterrows():
                    fpath = row[target_col]
                    img = _IMG_PIXEL_CACHE.get(fpath)
                    if img is None:
                        img = cv2.imread(fpath)
                        if img is None:
                            raise ValueError(f"cv2.imread failed for: {fpath}")
                        img = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
                        _IMG_PIXEL_CACHE[fpath] = img
                    X_lst.append(img)
                x_np = np.asarray(X_lst)
                datasets_train_dict[file_name]['X'] = x_np
                datasets_train_dict[file_name]['Y'] = class_labels
            return datasets_train_dict
        else:
            AssertionError(f'Filepath {self.datasets_path} did not exists!!')

    def prepare_chunks(self, df_dict, class_label="class"):
        out_dict = {
            data_name.split('/')[-1].split('.')[0].upper(): {'X': None, 'Y': None}
            for data_name, _ in df_dict.items()
        }
    
        for data_name, df in df_dict.items():
            data_name = data_name.split('/')[-1].split('.')[0].upper()
            class_labels = df.loc[:, df.columns.str.startswith(class_label)]
    
            if self.use_img and not self.use_embeds:
                X_lst = []
                target_col = next(
                    (c for c in ['file_path', 'filepath', 'file', 'filename']
                    if c in df.columns), None)
                if target_col is None:
                    raise KeyError(
                        "Image DataFrame must contain a file path column "
                        "('file_path', 'filepath', 'file', or 'filename')")
                for idx, row in df.iterrows():
                    fpath = row[target_col]
                    img = _IMG_PIXEL_CACHE.get(fpath)
                    if img is None:
                        img = cv2.imread(fpath)
                        if img is None:
                            raise ValueError(f"cv2.imread failed for: {fpath}")
                        img = np.transpose(img, (2, 0, 1))   # (H,W,C) -> (C,H,W)
                        _IMG_PIXEL_CACHE[fpath] = img
                    X_lst.append(img)
                out_dict[data_name]['X'] = np.asarray(X_lst)
    
            else:
                target_col = next((c for c in ['file_path', 'filepath', 'file', 'filename'] if c in df.columns), None)
    
                embed_cols = [c for c in df.columns if c.startswith('X_') and c[2:].isdigit()]
                is_embedded_df = (len(embed_cols) > 0 and target_col is None)
    
                if is_embedded_df:
                    row_tuples = df[embed_cols]
                else:
                    # Standard tabular or image-embed case
                    row_tuples = df.drop(columns=list(class_labels.columns))
                    if self.use_embeds:
                        cols_to_drop = []
                        if target_col is not None:
                            cols_to_drop.append(target_col)
                        if (isinstance(self.fairness_attr, str) and self.fairness_attr in df.columns):
                            cols_to_drop.append(self.fairness_attr)
                        if cols_to_drop:
                            row_tuples = df.drop(columns=list(class_labels.columns) + cols_to_drop)
    
                out_dict[data_name]['X'] = row_tuples
    
            out_dict[data_name]['Y'] = class_labels
    
        return out_dict
    

    def save_data_chunks(self, file_dict, filename):
        out_path_dir = os.path.join(self.outpath, filename)
        self.datasets_path = out_path_dir

        if not os.path.exists(out_path_dir):
            print(f'Create new directory {out_path_dir}')
            os.makedirs(out_path_dir, exist_ok=True)

        dataset_filenames = []

        for file_name, df_ in file_dict.items():
            file_path = os.path.join(out_path_dir, file_name)

            df_.to_csv(file_path, index=False)
            dataset_filenames.append(file_path)
            self.datasets_filenames[file_name.split('.')[0].upper()] = file_path
    
    def split_pandas_ordered(self, data, chunk_size):
        chunk_indices = np.array_split(np.arange(len(data)), chunk_size)
        return [data.iloc[idxs].copy() for idxs in chunk_indices]

    def convert_data_with_pca(self, data, k_dim=3):
        pca = PCA(n_components=k_dim)
        data_tuples_embds = pca.fit_transform(data)
        return data_tuples_embds
    
    def convert_text_data_to_vectors(self, data, model_name='hashing'):
        if isinstance(data, dict):
            filenames = list(data.keys())
            class_lst = [None] * len(filenames)
        elif isinstance(data, str):
            # Must be a real CSV path
            csv_path = (data if os.path.isabs(data)
                        else os.path.join(os.path.dirname(
                            os.path.abspath(self.datasets_path)), data))
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"convert_text_data_to_vectors received '{data}' which does not resolve to an existing file. Pass a DataFrame instead.")
            df_src = pd.read_csv(csv_path)
            target_col = next((c for c in ['file_path', 'filepath', 'file', 'filename'] if c in df_src.columns), None)
            if target_col is None:
                raise KeyError("No file column found in CSV.")
            filenames = df_src[target_col].tolist()
            class_lst = df_src[self.label_class].tolist()
        elif isinstance(data, pd.DataFrame):
            target_col = next((c for c in ['file_path', 'filepath', 'file', 'filename'] if c in data.columns), None)
            if target_col is None:
                raise KeyError("No file column found in DataFrame.")
            filenames = data[target_col].tolist()
            class_lst = data[self.label_class].tolist()
        else:
            raise TypeError(
                f"convert_text_data_to_vectors: unsupported type {type(data)}")
    
        if model_name == 'hashing':
            texts = []
            empty_count = 0
            missing_count = 0
    
            for filename in tqdm(filenames, desc="Reading text files"):
                if not os.path.exists(str(filename)):
                    missing_count += 1
                    texts.append(' ')
                    continue
                try:
                    with open(filename, encoding='utf-8', errors='ignore') as f:
                        txt_ = f.read().strip()
                except OSError:
                    txt_ = ''
                if not txt_:
                    empty_count += 1
                    txt_ = ' '
                texts.append(txt_)
    
            if missing_count > 0:
                print(f"  WARNING: {missing_count}/{len(filenames)} text files not found, their vectors will be zero. Check update_img_fullpath resolution.")
            if empty_count > 0:
                print(f"  WARNING: {empty_count}/{len(filenames)} text files were empty.")
    
            vectorizer = HashingVectorizer(n_features=768, alternate_sign=False, norm='l2')
            X_embed_lst = vectorizer.transform(texts).toarray().astype(np.float64)
    
            out_dict = {fn: X_embed_lst[i] for i, fn in enumerate(filenames)}
            out_df = pd.DataFrame(X_embed_lst, columns=[f"X_{i+1}" for i in range(X_embed_lst.shape[1])])
            out_df[self.label_class] = class_lst
            return out_dict, out_df
    
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        if model_name == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            lm = GPT2Model.from_pretrained(model_name, output_hidden_states=True).to(device)
            lm.eval()
            tokenizer.pad_token = tokenizer.eos_token
            lm.config.pad_token_id = tokenizer.eos_token_id
        elif model_name == 'lamma':
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            lm = AutoModel.from_pretrained(model_id, output_hidden_states=True).to(device)
            lm.eval()
        elif model_name == 'bert':
            model_id = "bert-base-uncased"
            tokenizer = BertTokenizer.from_pretrained(model_id)
            lm = BertModel.from_pretrained(model_id, output_hidden_states=True).to(device)
            lm.eval()
        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")
    
        out_dict = {}
        X_embed_lst = []
    
        for idx, filename in tqdm(enumerate(filenames), total=len(filenames), desc="Embedding text"):
            if not os.path.exists(str(filename)):
                print(f"  WARNING: missing file {filename} — using empty string.")
                txt_ = ' '
            else:
                try:
                    with open(filename, encoding='utf-8', errors='ignore') as f:
                        txt_ = f.read().strip() or ' '
                except OSError:
                    txt_ = ' '
            inputs = tokenizer([txt_], padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = lm(**inputs)
                if model_name == 'bert':
                    emb = outputs.last_hidden_state[:, 0, :].flatten()
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    emb = ((outputs.last_hidden_state * mask).sum(1) / mask.sum(1)).flatten()
                emb = emb.detach().cpu().numpy()
    
            out_dict[filename] = emb
            X_embed_lst.append(emb)
    
        columns = [f"X_{i+1}" for i in range(len(X_embed_lst[0]))]
        out_df = pd.DataFrame(X_embed_lst, columns=columns)
        out_df[self.label_class] = class_lst
        return out_dict, out_df

    def convert_text_data_to_lstm_inputs(self, data, max_vocab_size=20000, max_len=512):
        target_col = None
        for col in ['file_path', 'file', 'filename']:
            if col in data.columns:
                target_col = col
                break

        if target_col is None:
            raise KeyError("Dataframe must contain a column named 'file_path', 'file', or 'filename'")

        tokenized_texts = []
        vocab_counter = Counter()
        for filename in tqdm(data[target_col].tolist()):
            with open(filename, encoding='utf-8', errors='ignore') as f:
                txt_ = f.read()
            tokens = re.findall(r"\b\w+\b", txt_.lower())
            if not tokens:
                tokens = ['']
            tokenized_texts.append(tokens)
            vocab_counter.update(tokens)

        vocab = {token: idx + 2 for idx, (token, _) in enumerate(vocab_counter.most_common(max_vocab_size - 2))}
        token_ids = np.zeros((len(tokenized_texts), max_len), dtype=np.int64)
        for i, tokens in enumerate(tokenized_texts):
            encoded = [vocab.get(token, 1) for token in tokens[:max_len]]
            token_ids[i, :len(encoded)] = encoded

        out_df = pd.DataFrame(token_ids, columns=[f"token_{i+1}" for i in range(max_len)])
        class_labels = data.loc[:, data.columns.str.startswith(self.label_class)]
        for col in class_labels.columns:
            out_df[col] = class_labels[col].values

        return token_ids, out_df

    def _get_clip_model_and_processor(self, device):
        cached = _CLIP_MODEL_CACHE.get(device)
        if cached is not None:
            return cached
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True, local_files_only=True)
        except Exception:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        model.eval()
        _CLIP_MODEL_CACHE[device] = (model, processor)
        return model, processor

    def _compute_base_img_txt_embeddings(self, img_data, y_labels, batch_size=256):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = self._get_clip_model_and_processor(device)

        if isinstance(img_data, dict):
            filenames = list(img_data.keys())
        elif isinstance(img_data, str):
            df = pd.read_csv(os.path.join(self.datasets_path, img_data))
            filenames = df.get('file_path', df.get('filename')).tolist()
        elif isinstance(img_data, pd.DataFrame):
            filenames = img_data.get('file_path', img_data.get('filename')).tolist()

        num_samples = len(filenames)
        all_img, all_txt = [], []
        for i in range(0, num_samples, batch_size):
            batch_files = filenames[i:i + batch_size]

            batch_labels = []
            for f in batch_files:
                val = self.idx_to_label[y_labels[f]]
                if isinstance(val, (list, np.ndarray, pd.Series)):
                    batch_labels.append(str(val[0]))
                else:
                    batch_labels.append(str(val))

            batch_imgs = [Image.open(f).convert("RGB") for f in batch_files]

            inputs = processor(
                text=batch_labels,
                images=batch_imgs,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                all_img.append(outputs.image_embeds.cpu())
                all_txt.append(outputs.text_embeds.cpu())

        all_img = torch.cat(all_img, dim=0).numpy().astype(np.float64)
        all_txt = torch.cat(all_txt, dim=0).numpy().astype(np.float64)

        img_emb_dict = {str(f): all_img[i] for i, f in enumerate(filenames)}
        txt_emb_dict = {str(f): all_txt[i] for i, f in enumerate(filenames)}
        return img_emb_dict, txt_emb_dict

    def _compute_fair_txt_embeddings(self, fair_feat, filenames, batch_size=256):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = self._get_clip_model_and_processor(device)

        if isinstance(fair_feat, (pd.DataFrame, pd.Series)):
            fair_vals = np.asarray(fair_feat).flatten()
        elif isinstance(fair_feat, np.ndarray):
            fair_vals = fair_feat.flatten()
        else:
            fair_vals = list(fair_feat)
        fair_texts = [str(feat) for feat in fair_vals]

        num_samples = len(filenames)
        all_fair = []
        for i in range(0, num_samples, batch_size):
            batch_fair = fair_texts[i:i + batch_size]
            f_inputs = processor(text=batch_fair, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                fair_emb = model.get_text_features(**f_inputs)
                if hasattr(fair_emb, 'pooler_output'):
                    fair_emb = fair_emb.pooler_output
                all_fair.append(fair_emb.cpu())

        all_fair = torch.cat(all_fair, dim=0).numpy().astype(np.float64)
        return {str(f): all_fair[i] for i, f in enumerate(filenames)}

    def _get_or_compute_img_embeddings(self, img_data, y_labels, fair_attrbs_raw_txt):
        base_key = self.datasets_path
        base_cached = _IMG_BASE_EMBED_CACHE.get(base_key)
        if base_cached is not None:
            print(f'Reusing cached CLIP base (image+label) embeddings for {self.datasets_path} '
                  f'-- skipping recomputation.')
            img_emb_dict, txt_emb_dict = base_cached
        else:
            start_time = time.time()
            img_emb_dict, txt_emb_dict = self._compute_base_img_txt_embeddings(img_data=img_data, y_labels=y_labels)
            print(f'Finish base (image+label) embedding with CLIP in {time.time() - start_time:.2f} seconds')
            _IMG_BASE_EMBED_CACHE[base_key] = (img_emb_dict, txt_emb_dict)

        fair_emb_dict = None
        if self.use_fairness and fair_attrbs_raw_txt is not None:
            fair_key = (self.datasets_path, self.fairness_attr)
            fair_cached = _IMG_FAIR_TXT_EMBED_CACHE.get(fair_key)
            if fair_cached is not None:
                print(f'Reusing cached CLIP fairness-text embeddings for {self.datasets_path} '
                      f'(fairness_attr={self.fairness_attr}) -- skipping recomputation.')
                fair_emb_dict = fair_cached
            else:
                start_time = time.time()
                fair_emb_dict = self._compute_fair_txt_embeddings(
                    fair_feat=fair_attrbs_raw_txt, filenames=list(img_emb_dict.keys()))
                print(f'Finish fairness-text embedding with CLIP in {time.time() - start_time:.2f} seconds')
                _IMG_FAIR_TXT_EMBED_CACHE[fair_key] = fair_emb_dict

        out_dict = {}
        for f, img_v in img_emb_dict.items():
            if fair_emb_dict is not None:
                out_dict[f] = (img_v + txt_emb_dict[f] + fair_emb_dict[f]) / 3
            else:
                out_dict[f] = (img_v + txt_emb_dict[f]) / 2

        return out_dict, img_emb_dict

    def convert_img_data_to_vectors(self, data, y_labels, fair_feat=None, allign_method='mean', batch_size=256):
        out_dict = {}
        out_dict_image_ebds = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True, local_files_only=True)
        except Exception:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        model.eval()

        if isinstance(data, dict):
            filenames = list(data.keys())
        elif isinstance(data, str):
            df = pd.read_csv(os.path.join(self.datasets_path, data))
            filenames = df.get('file_path', df.get('filename')).tolist()
        elif isinstance(data, pd.DataFrame):
            filenames = data.get('file_path', data.get('filename')).tolist()
        
        fair_texts = None
        if self.use_fairness and fair_feat is not None:
            if isinstance(fair_feat, (pd.DataFrame, pd.Series)):
                fair_vals = np.asarray(fair_feat).flatten()
            elif isinstance(fair_feat, np.ndarray):
                fair_vals = fair_feat.flatten()
            else:
                fair_vals = list(fair_feat)
            fair_texts = [str(feat) for feat in fair_vals]

        num_samples = len(filenames)
        all_fused = []
        all_images = []


        for i in range(0, num_samples, batch_size):
            batch_files = filenames[i : i + batch_size]
            
            batch_labels = []
            for f in batch_files:
                
                val = self.idx_to_label[y_labels[f]]
                if isinstance(val, (list, np.ndarray, pd.Series)):
                    batch_labels.append(str(val[0]))
                else:
                    batch_labels.append(str(val))

            batch_imgs = [Image.open(f).convert("RGB") for f in batch_files]

            inputs = processor(
                text=batch_labels,
                images=batch_imgs,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                img_emb = outputs.image_embeds 
                txt_emb = outputs.text_embeds   

                fair_emb = None
                if fair_texts:
                    batch_fair = fair_texts[i : i + batch_size]
                    f_inputs = processor(text=batch_fair, return_tensors="pt", padding=True).to(device)
                    fair_emb = model.get_text_features(**f_inputs)
                    if hasattr(fair_emb, 'pooler_output'):
                        fair_emb = fair_emb.pooler_output

                if allign_method == 'mean':
                    if fair_emb is not None:
                        fused = (img_emb + txt_emb + fair_emb) / 3
                    else:
                        fused = (img_emb + txt_emb) / 2
                elif allign_method == 'concat':
                    if fair_emb is not None:
                        fused = torch.cat([img_emb, txt_emb, fair_emb], dim=-1)
                    else:
                        fused = torch.cat([img_emb, txt_emb], dim=-1)

                all_fused.append(fused.cpu())
                all_images.append(img_emb.cpu())

        all_fused = torch.cat(all_fused, dim=0).numpy().astype(np.float64)
        all_images = torch.cat(all_images, dim=0).numpy().astype(np.float64)
        
        for i, file_path in enumerate(filenames):
            safe_key = str(file_path)
            out_dict[safe_key] = all_fused[i]
            out_dict_image_ebds[safe_key] = all_images[i]

        return out_dict, out_dict_image_ebds
    

    def split_by_cluster(self, chunk_size, data_tuples_embds):
        target_num_chunks = int(np.ceil(data_tuples_embds.shape[0] / chunk_size))

        clustering = KMeans(n_clusters=target_num_chunks, random_state=42, algorithm="lloyd", n_init=10)
        initial_labels = clustering.fit_predict(data_tuples_embds)

        chunks = {}
        cluster_id = 0
        
        for label in np.unique(initial_labels):
            indices = np.where(initial_labels == label)[0]
            
            cluster_data = data_tuples_embds[indices]
            if len(indices) <= chunk_size:
               chunks[cluster_id] = indices
               cluster_id += 1 
            else:
                num_of_subclusters = int(np.ceil(len(indices) / chunk_size))
                if num_of_subclusters <= 1:  
                    num_of_subclusters = 2
                
                sub_clustering = KMeans(n_clusters=num_of_subclusters, random_state=42, n_init=10)
                sub_labels = sub_clustering.fit_predict(cluster_data)

                for sub_label in np.unique(sub_labels):
                    sub_indices = indices[sub_labels == sub_label]
                    chunks[cluster_id] = sub_indices
                    cluster_id += 1

        singletons = [cid for cid, idxs in chunks.items() if len(idxs) == 1]
        for singleton_cid in singletons:
            # Check if it still exists (might have been deleted in previous iterations)
            if singleton_cid not in chunks: 
                continue
                
            singleton_idx = chunks[singleton_cid][0]
            singleton_vec = data_tuples_embds[singleton_idx].reshape(1, -1)

            candidate_clusters = {cid: idxs for cid, idxs in chunks.items() if cid != singleton_cid}
            
            if not candidate_clusters: # Edge case: no other clusters exist
                break 

            centroids = np.array([data_tuples_embds[idxs].mean(axis=0) for idxs in candidate_clusters.values()])

            closest_order = np.argsort(np.linalg.norm(centroids - singleton_vec, axis=1))

            merged = False
            for order in closest_order:
                target_cid = list(candidate_clusters.keys())[order]
                if len(chunks[target_cid]) < chunk_size:
                    chunks[target_cid] = np.append(chunks[target_cid], singleton_idx)
                    merged = True
                    break

            if merged:
                del chunks[singleton_cid]
        
        
        while len(chunks) < target_num_chunks:
            # Find the ID of the chunk with the most elements
            largest_cid = max(chunks.keys(), key=lambda k: len(chunks[k]))
            largest_idxs = chunks[largest_cid]
            
            if len(largest_idxs) <= 1:
                break
                
            # Split the largest chunk into 2 subclusters
            cluster_data = data_tuples_embds[largest_idxs]
            sub_clustering = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_labels = sub_clustering.fit_predict(cluster_data)
            
            sub_idx_0 = largest_idxs[sub_labels == 0]
            sub_idx_1 = largest_idxs[sub_labels == 1]
            
            if len(sub_idx_0) == 0 or len(sub_idx_1) == 0:
                midpoint = len(largest_idxs) // 2
                sub_idx_0 = largest_idxs[:midpoint]
                sub_idx_1 = largest_idxs[midpoint:]
            
            chunks[largest_cid] = sub_idx_0
            
            new_cid = max(chunks.keys()) + 1
            chunks[new_cid] = sub_idx_1

        return chunks


    def split_by_tree(self, chunk_size, data_tuples_embds, use_query_r=False):
        N = data_tuples_embds.shape[0]

        # Expected number of chunks — used to validate output
        expected_chunks = max(1, int(np.ceil(N / chunk_size)))

        tree = KDTree(data_tuples_embds, leaf_size=40)

        visited = np.zeros(N, dtype=bool)
        chunks_idx = {}
        chunk_id = 0

        k_search = min(N, chunk_size)

        for i in range(N):
            if visited[i]:
                continue

            query_tuple = data_tuples_embds[i:i+1]

            if use_query_r:
                rad_ = 5
                indxs = tree.query_radius(query_tuple, r=rad_)[0]
                valid_mask = ~visited[indxs]
                candidates = indxs[valid_mask]
                if len(candidates) > chunk_size:
                    candidates = candidates[:chunk_size]
            else:
                dists, indxs = tree.query(query_tuple, k=k_search)
                indxs = indxs[0]
                dists = dists[0]
                valid_mask = ~visited[indxs]
                candidates = indxs[valid_mask]

                if len(candidates) > chunk_size:
                    candidates = candidates[:chunk_size]

            if len(candidates) > 0:
                chunks_idx[chunk_id] = candidates.tolist()
                visited[candidates]  = True
                chunk_id += 1
        min_limit = 3
        orphans= []
        ids_to_remove  = []
        point_to_chunk_map = {}

        for cid, idxs in chunks_idx.items():
            if len(idxs) < min_limit:
                orphans.extend(idxs)
                ids_to_remove.append(cid)
            else:
                for idx in idxs:
                    point_to_chunk_map[idx] = cid

        for cid in ids_to_remove:
            del chunks_idx[cid]

        for orphan_idx in orphans:
            orphan_vec = data_tuples_embds[orphan_idx].reshape(1, -1)
            _, neighbor_idxs = tree.query(orphan_vec, k=min(N, 50))
            neighbor_idxs = neighbor_idxs[0]

            merged = False
            for n_idx in neighbor_idxs:
                if n_idx == orphan_idx:
                    continue
                if n_idx in point_to_chunk_map:
                    target_cid = point_to_chunk_map[n_idx]
                    if len(chunks_idx[target_cid]) < chunk_size:
                        chunks_idx[target_cid].append(orphan_idx)
                        point_to_chunk_map[orphan_idx] = target_cid
                        merged = True
                        break

            if not merged:
                for n_idx in neighbor_idxs:
                    if n_idx == orphan_idx:
                        continue
                    if n_idx in point_to_chunk_map:
                        target_cid = point_to_chunk_map[n_idx]
                        chunks_idx[target_cid].append(orphan_idx)
                        point_to_chunk_map[orphan_idx] = target_cid
                        merged = True
                        break

        actual_chunks = len(chunks_idx)
        if actual_chunks < expected_chunks // 2:
            print(f"  WARNING split_by_tree: expected ~{expected_chunks} chunks "
                  f"but got {actual_chunks}. "
                  f"N={N}, chunk_size={chunk_size}, "
                  f"data_dim={data_tuples_embds.shape[1]}. "
                  f"Consider using split_by='cluster' for low-dimensional inputs.")

        return chunks_idx


    def split_by_tree_fair_tabular(self, chunk_size, data_tuples_embds):
        chunks_idx = {}
        idxs_ = [i for i in range(data_tuples_embds.shape[0])]
        data_tuples_embds_tmp = data_tuples_embds
        chunk_id = 0
        for i in range(data_tuples_embds.shape[0]):
            flag = True
            for idxs in chunks_idx.values():
                if i in idxs:
                    flag = False

            if flag:
                tree = KDTree(data_tuples_embds_tmp, leaf_size=6)
                query_tuple = data_tuples_embds[i, :].reshape(-1, data_tuples_embds[i, :].shape[0])
                if chunk_size <= len(idxs_):
                    dist, indxs = tree.query(query_tuple, k=chunk_size)
                else:
                    chunks_idx[chunk_id] = idxs_
                    return chunks_idx
                
                chunks_idx[chunk_id] = [idxs_[i] for i in indxs[0]]
                data_tuples_embds_tmp = np.delete(data_tuples_embds_tmp, indxs[0], axis=0)
                idxs_ = [j for i, j in enumerate(idxs_) if i not in indxs[0]]

                chunk_id+=1

        return chunks_idx



    def update_img_fullpath(self, df):
 
        target_col = next((c for c in ['file_path', 'filepath', 'file', 'filename'] if c in df.columns), None)
    
        if target_col is None:
            raise KeyError("DataFrame must contain a file path column ('file_path', 'filepath', 'file', or 'filename')")
    
        datasets_dir = os.path.dirname(os.path.abspath(self.datasets_path))
        parent_dir = os.path.dirname(datasets_dir)
        grandparent = os.path.dirname(parent_dir)
    
        missing_count = 0
        hint = None

        def resolve(path_value):
            nonlocal missing_count, hint
            raw = str(path_value).strip()

            if os.path.isabs(raw) and os.path.exists(raw):
                return raw

            if os.path.isabs(raw):
                parts = [p for p in raw.split(os.sep) if p]
                if hint is not None:
                    start, base = hint
                    if start < len(parts):
                        c = os.path.normpath(os.path.join(base, *parts[start:]))
                        if os.path.exists(c):
                            return c
                for start in range(1, len(parts)):
                    for base in (datasets_dir, parent_dir, grandparent):
                        c = os.path.normpath(os.path.join(base, *parts[start:]))
                        if os.path.exists(c):
                            hint = (start, base)
                            return c

            c = os.path.normpath(os.path.join(datasets_dir, raw))
            if os.path.exists(c):
                return c
    
            c = os.path.normpath(os.path.join(parent_dir, raw))
            if os.path.exists(c):
                return c
    
            c = os.path.normpath(os.path.join(grandparent, raw))
            if os.path.exists(c):
                return c
    
            basename = os.path.basename(raw)
            c = os.path.join(datasets_dir, basename)
            if os.path.exists(c):
                return c
    
            c = os.path.join(parent_dir, basename)
            if os.path.exists(c):
                return c
    
            missing_count += 1
            return os.path.normpath(os.path.join(datasets_dir, raw))
    
        df[target_col] = df[target_col].apply(resolve)
    
        if missing_count > 0:
            print(f"  WARNING: {missing_count}/{len(df)} paths in '{target_col}' could not be resolved. Check your file column and dataset_path.")
            print(f"  datasets_path = {self.datasets_path}")
            print(f"  Example unresolved: {df[target_col].iloc[0]}")
    
        return df
    
    def calculate_num_of_classes(self, df, class_label="class"):
        class_labels = df.loc[:, df.columns.str.startswith(class_label)]
        class_unique = class_labels.drop_duplicates()[class_label].tolist()
        self.num_of_classes = len(class_unique)


    def labels_to_idx(self, df, class_label="class", idx_to_label=None):
        class_labels = df.loc[:, df.columns.str.startswith(class_label)]
        
        class_unique = class_labels.drop_duplicates()[class_label].tolist()
        
        if idx_to_label is None:
            self.label_to_idx = {label: i for i, label in enumerate(class_unique)}
            self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        elif len(idx_to_label) == 0:
            self.label_to_idx = {label: i for i, label in enumerate(class_unique)}
            self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        else:
            self.idx_to_label = idx_to_label
            self.label_to_idx = {label: i for i, label in self.idx_to_label.items()}

        self.num_of_classes = len(class_unique)


    def get_fair_attrbs_(self, df_):
        if all(isinstance(i, int) for i in self.fairness_attr):
            cols = df_.columns[self.fairness_attr]
        else:
            cols = self.fairness_attr
        res = df_.loc[:, cols]

        if isinstance(res, pd.Series):
            return res.to_frame()
        
        return res


    def create_group_labels(self, df_, task_label_col='class'):
        sensitive_df = self.get_fair_attrbs_(df_)
        
        attr_names = [task_label_col] + list(sensitive_df.columns)

        combined = df_[attr_names].astype(str).agg(', '.join, axis=1)
        unique_combinations = sorted(combined.unique())

        group_mapping = {combo: i for i, combo in enumerate(unique_combinations)}
        
        group_ret = combined.map(group_mapping).astype(int)
        group_ret.name = "group_id"

        print("Fairness Groupping Mapping Legend: ")
        for combo, idx in group_mapping.items():
            print(f"Group {idx}: ({combo})")
        print(f"Total Groups: {len(unique_combinations)}")
        return group_ret

    def split_data(self, chunk_size, split_by='ordered', save_it=False, data_type='tabular', convert_labels=False, X_embeds=None, idx_to_label=None):
        if os.path.exists(self.datasets_path):
            out_dict = {}
            out_dict_fair_groups = {}
            if data_type.lower() == 'image' or data_type.lower() == 'img':
                self.use_img = True
                df_img_info = pd.read_csv(self.datasets_path, index_col=False)
                df_img_info = self.update_img_fullpath(df=df_img_info)
                file_name = self.datasets_path.split('/')[-1].split('.')[0].upper()
                self.calculate_num_of_classes(df=df_img_info, class_label=self.label_class)

                target_col = None
                for col in ['file_path', 'file', 'filename']:
                    if col in df_img_info.columns:
                        target_col = col
                        break
                
                if convert_labels:
                    self.labels_to_idx(df=df_img_info, idx_to_label=idx_to_label, class_label=self.label_class)
                    col = df_img_info[self.label_class]
                    mapped = col.map(self.label_to_idx)
                    mapped = mapped.fillna(col.where(col.isin(list(self.idx_to_label.keys()))))
                    if mapped.isna().any():
                        bad = col[mapped.isna()].unique()[:10]
                        raise ValueError(
                            f"'{self.label_class}' values not found in label_to_idx "
                            f"({list(self.label_to_idx.keys())[:10]}...): {list(bad)}")
                    df_img_info[self.label_class] = mapped
                
                if idx_to_label is not None:
                    self.labels_to_idx(df=df_img_info, idx_to_label=idx_to_label, class_label=self.label_class)
                
                fair_attrbs_groups = None
                if self.use_fairness:
                    fair_attrbs_raw_txt = self.get_fair_attrbs_(df_=df_img_info,)
                    fair_attrbs_groups = self.create_group_labels(df_=df_img_info, task_label_col=self.label_class)
                    self.num_of_groups = np.unique(fair_attrbs_groups).shape[0]
                else:
                    fair_attrbs_raw_txt = None
                    fair_attrbs_groups = None
                
                X_embds_ret=None
                Y_labels = {row[target_col]: row[self.label_class] for index, row in df_img_info.iterrows()}
                if split_by == 'ordered':
                    start_time = time.time()
                    df_chunks = self.split_pandas_ordered(df_img_info, chunk_size)
                    if self.use_fairness:
                        df_fairs_ret = self.split_pandas_ordered(fair_attrbs_groups, chunk_size)

                    print(f'Finish splitting by ordered in {time.time() - start_time:.2f} seconds')

                elif split_by.lower() == 'random':
                    start_time = time.time()
                    perm = np.random.permutation(len(df_img_info))
                    perm_chunks = np.array_split(perm, chunk_size)
                    df_chunks = [df_img_info.iloc[idxs].copy() for idxs in perm_chunks]
                    if self.use_fairness:
                        df_fairs_ret = [fair_attrbs_groups.iloc[idxs] for idxs in perm_chunks]
                    print(f'Finish splitting by random in {time.time() - start_time:.2f} seconds')

                elif split_by.lower() == 'tree':
                    img_data = (None if self.datasets_path in _IMG_BASE_EMBED_CACHE else
                               {row[target_col]: None for index, row in df_img_info.iterrows()})
                    X_embds, X_embeds_img = self._get_or_compute_img_embeddings(img_data, Y_labels, fair_attrbs_raw_txt)

                    X_embds = [embds for path_, embds in X_embds.items()]
                    X_embds = np.asarray(X_embds, dtype=np.float64)
                    start_time = time.time()
                    chunk_idxs = self.split_by_tree(data_tuples_embds=X_embds, chunk_size=chunk_size)
                    print(f'Finish splitting by tree in {time.time() - start_time:.2f} seconds')
                    X_embeds_img = [embds for path_, embds in X_embeds_img.items()]
                    X_embeds_img = np.asarray(X_embeds_img, dtype=np.float64)
                    if self.use_fairness:
                        df_fairs_ret = [fair_attrbs_groups.iloc[idxs] for idxs in chunk_idxs.values()]
                    if self.use_embeds:
                        columns = [f"X_{i+1}" for i in range(len(X_embeds_img[0]))]
                        # Build dataframe
                        df_embeds = pd.DataFrame(X_embeds_img, columns=columns)

                        # Add class column
                        df_embeds[self.label_class] = df_img_info[self.label_class].to_list()
                        df_embeds[target_col] = df_img_info[target_col].to_list()
                        if self.use_fairness:
                            df_embeds[self.fairness_attr] = fair_attrbs_raw_txt
                        elif isinstance(self.fairness_attr, str) and self.fairness_attr in df_img_info.columns:
                            df_embeds[self.fairness_attr] = df_img_info[self.fairness_attr].to_list()
                        df_chunks = [df_embeds.iloc[idxs] for idxs in chunk_idxs.values()]
                    else:
                        df_chunks = [df_img_info.iloc[idxs] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embeds_img[idxs] for idxs in chunk_idxs.values()]
                elif split_by.lower() == 'cluster':
                    # Same reasoning as the 'tree' branch above.
                    img_data = (None if self.datasets_path in _IMG_BASE_EMBED_CACHE else
                               {row[target_col]: None for index, row in df_img_info.iterrows()})
                    X_embds, X_embeds_img = self._get_or_compute_img_embeddings(img_data, Y_labels, fair_attrbs_raw_txt)
                    X_embds = [embds for path_, embds in X_embds.items()]
                    X_embds = np.asarray(X_embds, dtype=np.float64)
                    start_time = time.time()
                    chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=X_embds)
                    print(f'Finish splitting by cluster in {time.time() - start_time:.2f} seconds')
                    X_embeds_img = [embds for path_, embds in X_embeds_img.items()]
                    X_embeds_img = np.asarray(X_embeds_img, dtype=np.float64)
                    if self.use_fairness:
                        df_fairs_ret = [fair_attrbs_groups.iloc[idxs] for idxs in chunk_idxs.values()]

                    if self.use_embeds:
                        columns = [f"X_{i+1}" for i in range(len(X_embeds_img[0]))]
                        # Build dataframe
                        df_embeds = pd.DataFrame(X_embeds_img, columns=columns)

                        # Add class column
                        df_embeds[self.label_class] = df_img_info[self.label_class].to_list()
                        df_embeds[target_col] = df_img_info[target_col].to_list()
                        if self.use_fairness:
                            df_embeds[self.fairness_attr] = fair_attrbs_raw_txt
                        elif isinstance(self.fairness_attr, str) and self.fairness_attr in df_img_info.columns:
                            # Not used by FLA-SH here (prepare_chunks drops it from X), but
                            # fairness evaluation of the selected chunks reads it back from
                            # the chunk CSVs -- e.g. the plain-SR arm of
                            # run_sr_vs_purity_sr_fairness.py.
                            df_embeds[self.fairness_attr] = df_img_info[self.fairness_attr].to_list()
                        df_chunks = [df_embeds.iloc[idxs] for idxs in chunk_idxs.values()]
                    else:
                        df_chunks = [df_img_info.iloc[idxs] for idxs in chunk_idxs.values()]
                    
                    X_embds_ret = [X_embeds_img[idxs] for idxs in chunk_idxs.values()]

                file_dict = {f'{file_name}_{i}.csv': df_ for i, df_ in enumerate(df_chunks)}
                if save_it:
                    self.save_data_chunks(file_dict=file_dict, filename=f'{file_name}_{chunk_size}')

                if self.use_fairness:
                    out_dict_fair_groups[file_name] = df_fairs_ret
                out_dict[file_name] = file_dict
                return out_dict, X_embds_ret, out_dict_fair_groups
            elif data_type == 'tabular':
                self.use_img = False
                files = glob.glob(self.datasets_path + "/*.csv")
                out_dict = {}
                out_dict_fair_groups = {}
                X_embds_ret=None
                for file in files:

                    df_chunks = []
                    df_fairs_ret = [] if self.use_fairness else None
                    df_file = drop_index_columns(pd.read_csv(file, index_col=False))
                    self.calculate_num_of_classes(df=df_file, class_label=self.label_class)
                    file_name = file.split('/')[-1].split('.')[0].upper()
                    X_embds_ret=None
                    df_file = df_file.fillna(-99)
                    
                    ## Find for each tabular dataset his fair attr
                    fair_attrbs_groups = None
                    if self.use_fairness:
                        fair_attrbs_groups = self.create_group_labels(df_=df_file, task_label_col=self.label_class)
                        self.num_of_groups = np.unique(fair_attrbs_groups).shape[0]

                    if split_by == 'ordered':
                        start_time = time.time()
                        df_chunks = self.split_pandas_ordered(df_file, chunk_size)
                        if self.use_fairness:
                            df_fairs_ret = self.split_pandas_ordered(fair_attrbs_groups, chunk_size)

                        print(f'Finish splitting by ordered in {time.time() - start_time:.2f} seconds')
                    elif split_by == 'date':
                        try:
                            pd.to_datetime(df_file['Date'], utc=False)
                            df_chunks = [group.drop(columns=['Date']) for date, group in df_file.groupby('Date')]
                        except Exception as ex:
                            pd.to_datetime(df_file['Date'], utc=False, format="mixed")
                            df_chunks = [group.drop(columns=['Date']) for date, group in df_file.groupby('Date')]
                    elif split_by == 'month':
                        pd.to_datetime(df_file['Date'], utc=False, format="mixed")
                        df_chunks = [group.drop(columns=['Date']) for date, group in df_file.groupby(pd.to_datetime(df_file['Date'],format='mixed').dt.month)]
                    elif split_by.lower() == 'tree':
                        print('Start KD-Tree...')
                        start_time = time.time()
                        if self.use_fairness:
                            class_labels_df = df_file.loc[:, df_file.columns.str.startswith('class')]
                            df_fair_input = pd.concat([fair_attrbs_groups, class_labels_df], axis=1)
                            _n_features = df_fair_input.shape[1]
                            _n_samples   = df_fair_input.shape[0]
                            _k_split = min(20, _n_features, _n_samples, max(2, _n_features))
                            X_embds_split = self.convert_data_with_pca(data=df_fair_input, k_dim=_k_split)
                            # 2D embedding kept for Hilbert ordering
                            _k_2d   = min(2, _n_features, _n_samples)
                            X_embds = self.convert_data_with_pca(data=df_fair_input, k_dim=_k_2d)
                        else:
                            _n_features = df_file.shape[1]
                            _n_samples   = df_file.shape[0]
                            _k_split = min(20, _n_features, _n_samples, max(2, _n_features))
                            X_embds_split = self.convert_data_with_pca(data=df_file, k_dim=_k_split)
                            _k_2d   = min(2, _n_features, _n_samples)
                            X_embds = self.convert_data_with_pca(data=df_file, k_dim=_k_2d)
                        print(f'Finish embedding with PCA in {time.time() - start_time:.2f} seconds')
                        start_time = time.time()
                        if self.use_fairness:
                            chunk_idxs = self.split_by_tree_fair_tabular(data_tuples_embds=X_embds_split, chunk_size=chunk_size)
                        else:
                            chunk_idxs = self.split_by_tree(data_tuples_embds=X_embds_split, chunk_size=chunk_size)
                        print(f'Finish splitting by tree in {time.time() - start_time:.2f} seconds')
                        df_chunks = [df_file.iloc[idxs] for idxs in chunk_idxs.values()]
                        if self.use_fairness:
                            df_fairs_ret = [fair_attrbs_groups.iloc[idxs] for idxs in chunk_idxs.values()]
                        X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                    elif split_by.lower() == 'cluster':
                        print('Start clustering...')
                        # Give fairness attrbs with their data to group it if fairness is selected otherwise the whole data
                        if self.use_fairness:
                            class_labels_df = df_file.loc[:, df_file.columns.str.startswith('class')]
                            df_fair_input = pd.concat([fair_attrbs_groups, class_labels_df], axis=1)
                            df_file_np = np.asarray(df_fair_input, dtype=np.float64)
                            start_time = time.time()
                            X_embds = self.convert_data_with_pca(data=df_file_np, k_dim=2)
                            print(f'Finish embedding with PCA in {time.time() - start_time:.2f} seconds')
                            start_time = time.time()
                            chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=df_file_np)
                            print(f'Finish splitting by cluster in {time.time() - start_time:.2f} seconds')
                        else:
                            df_file_np = np.asarray(df_file, dtype=np.float64)
                            # df_file_np = np.asarray(df_file, dtype=np.float64)
                            start_time = time.time()
                            if self.use_embeds:
                                X_embds = self.convert_data_with_pca(data=df_file_np, k_dim=3)
                                print(f'Finish embedding with PCA in {time.time() - start_time:.2f} seconds')
                                start_time = time.time()
                                chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=X_embds)
                                print(f'Finish splitting by cluster in {time.time() - start_time:.2f} seconds')
                            else:
                                chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=df_file_np)
                            print(f'Finish splitting by cluster in {time.time() - start_time:.2f} seconds')
                        df_chunks = [df_file.iloc[list(idxs)] for idxs in chunk_idxs.values()]
                        if self.use_fairness:
                            df_fairs_ret = [fair_attrbs_groups.iloc[idxs] for idxs in chunk_idxs.values()]
                        X_embds_ret = None

                    file_dict = {f'{file_name}_{i}.csv': df_ for i, df_ in enumerate(df_chunks)}

                    if save_it:
                        self.save_data_chunks(file_dict=file_dict, filename=f'{file_name}_{chunk_size}')
                    out_dict[file_name] = file_dict
                    if self.use_fairness:
                        out_dict_fair_groups[file_name] = df_fairs_ret

                # Make the attributes to return for each file
                return out_dict, X_embds_ret, out_dict_fair_groups
            elif data_type.lower() == 'text' or data_type.lower() == 'textual':
                out_dict_fair_groups = {}
                self.use_img = False
                df_text_info = pd.read_csv(self.datasets_path, index_col=False)
                df_text_info = self.update_img_fullpath(df=df_text_info)
                file_name = self.datasets_path.split('/')[-1].split('.')[0].upper()
                
                if self.problem == 'classification':
                    self.calculate_num_of_classes(df=df_text_info, class_label=self.label_class)

                if convert_labels:
                    self.labels_to_idx(df=df_text_info, class_label=self.label_class)
                    df_text_info[self.label_class] = df_text_info[self.label_class].map(self.label_to_idx)
                
                raw_text_info = df_text_info.copy()
                start_time = time.time()
                X_embds, embed_text_info = self.convert_text_data_to_vectors(data=df_text_info)
                print(f'Finish embedding with BERT in {time.time() - start_time:.2f} seconds')

                if self.model_type in ('LSTM', 'Text_LSTM', 'LSTM_Regression') and not self.use_embeds:
                    start_time = time.time()
                    _, df_text_info = self.convert_text_data_to_lstm_inputs(data=raw_text_info)
                    print(f'Finish tokenizing text for LSTM in {time.time() - start_time:.2f} seconds')
                else:
                    df_text_info = embed_text_info

                X_embds = [embds for _, embds in X_embds.items()]
                X_embds = np.asarray(X_embds, dtype=np.float64)
                if split_by == 'ordered':
                    start_time = time.time()
                    df_chunks = self.split_pandas_ordered(df_text_info, chunk_size)
                    X_embds_ret = np.array_split(X_embds, chunk_size)
                    print(f'Finish splitting by ordered in {time.time() - start_time:.2f} seconds')
                elif split_by.lower() == 'tree':
                    start_time = time.time()
                    chunk_idxs = self.split_by_tree(data_tuples_embds=X_embds, chunk_size=chunk_size)
                    print(f'Finish splitting by tree in {time.time() - start_time:.2f} seconds')
                    df_chunks = [df_text_info.iloc[idxs] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                elif split_by.lower() == 'cluster':
                    start_time = time.time()
                    chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=X_embds)
                    print(f'Finish splitting by cluster in {time.time() - start_time:.2f} seconds')
                    df_chunks = [df_text_info.iloc[list(idxs)] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                
                file_dict = {f'{file_name}_{i}.csv': df_ for i, df_ in enumerate(df_chunks)}
                if save_it:
                    self.save_data_chunks(file_dict=file_dict, filename=f'{file_name}_{chunk_size}')
                out_dict[file_name] = file_dict

                return out_dict, X_embds_ret, out_dict_fair_groups
            elif data_type.lower() == 'vectors' or data_type.lower() == 'vector embeddings':
                df_info = drop_index_columns(pd.read_csv(self.datasets_path, index_col=False))
                file_name = self.datasets_path.split('/')[-1].split('.')[0].upper()
                
                self.calculate_num_of_classes(df=df_info, class_label=self.label_class)
                X_embds_df = df_info.drop(columns=[self.label_class])
                X_embds = X_embds_df.values.tolist()
                X_embds = np.asarray(X_embds, dtype=np.float64)

                if split_by == 'ordered':
                        df_chunks = self.split_pandas_ordered(df_info, chunk_size)
                        X_embds_ret = np.array_split(X_embds, chunk_size)
                elif split_by.lower() == 'tree':
                    chunk_idxs = self.split_by_tree(data_tuples_embds=X_embds, chunk_size=chunk_size)
                    df_chunks = [df_info.iloc[idxs] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                elif split_by.lower() == 'cluster':
                    chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=X_embds)
                    df_chunks = [df_info.iloc[list(idxs)] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                
                file_dict = {f'{file_name}_{i}.csv': df_ for i, df_ in enumerate(df_chunks)}
                if save_it:
                    self.save_data_chunks(file_dict=file_dict, filename=f'{file_name}_{chunk_size}')
                out_dict[file_name] = file_dict

                return out_dict, X_embds_ret

            else:
                AssertionError(f'Wrong Data Type we did not support {data_type} data yet!!\nAvailable data types is the text, tabular, image, or give a csv file with vectors embeddings per data item')
        else:
            AssertionError('Error on reading file, did not exists')

    def read_data_split_chunk(self, chunk_size, save_it=False):
        if os.path.exists(self.datasets_path):
            filename = self.datasets_path.split('/')[-1].split('.')[0].upper()
            df = pd.read_csv(self.datasets_path)
            dict_df = {f'{filename}_{c}': df.iloc[i:i+chunk_size] for c, i in enumerate(range(0, df.shape[0],chunk_size))}
            if save_it:
                self.save_data_chunks(data_chunks=dict_df)
            return dict_df
        else:
            AssertionError('Error on reading file, did not exists')
            

    def save_output(self, method_name='fla_sh', ):
        if method_name.lower() == 'fla_sh':
            file_name = os.path.join(self.outpath, f'fla_sh.pickle')
        else:
            file_name = os.path.join(self.outpath, f'ds.pickle')
        with open(os.path.join(file_name), 'wb') as handle:
            
                pkl.dump(self.values_ds_FLA_SH_run, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def load_values(self, filepath, method_name='fla_sh'):
        with open(filepath, 'rb') as handle:
            if method_name.lower() == 'fla_sh':
                b = pkl.load(handle)
                self.values_ds_FLA_SH_run = b

    def get_score(self, X, y, use_img=False, use_NN=False, lr=0.0001, max_epochs=1, batch_size=1, val_frac=0.1, data_type='tabular', fair_attr_df=None):
        if data_type == 'text':
            use_norm = False
        else:
            use_norm = True

        if self.use_embeds and self.use_img:
            self.model = create_model(model_type=self.model_type,problem=self.problem, max_epochs=max_epochs, 
                                    batch_size=batch_size, lr=lr, val_frac=val_frac, hidden_layer_size=self.hidden_layers, 
                                    use_img=False, use_NN=use_NN, num_classes=self.num_of_classes, norm=use_norm, use_fairness=self.use_fairness, 
                                    fairness_attr=self.fairness_attr, num_of_fair=self.num_of_groups, lambda_fair=self.lambda_fair, fair_metric=self.fair_metric)
        else:
            self.model = create_model(model_type=self.model_type,problem=self.problem, max_epochs=max_epochs, 
                                    batch_size=batch_size, lr=lr, val_frac=val_frac, hidden_layer_size=self.hidden_layers, 
                                    use_img=use_img, use_NN=use_NN, num_classes=self.num_of_classes, norm=use_norm, use_fairness=self.use_fairness, 
                                    fairness_attr=self.fairness_attr, num_of_fair=self.num_of_groups, lambda_fair=self.lambda_fair, fair_metric=self.fair_metric)
        if use_NN and not use_img:
            self.model = self._configure_model(self.model)
        if use_img:
            self.model.fit(X, y.values.ravel(), X_val=None, y_val=False, use_val=False, fair_attrbs=fair_attr_df)
            X_torch = transform_to_torch(X).to(self.model.device)
            y_pred = self.model.predict(X_torch)   
        elif use_NN:
            self.model.fit(X, y.values.ravel(), X_val=None, y_val=False, use_val=False, fair_attrbs=fair_attr_df)
            X_in = self.model.feature_transform(X) if getattr(self.model, 'feature_transform', None) is not None else X
            X_torch = transform_to_torch(X_in).to(self.model.device)
            y_pred = self.model.predict(X_torch)            
        else:
            self.model.fit(X, y.values.ravel())
            y_pred = self.model.predict(X)
        
        if use_img or use_NN:
            y_pred = y_pred.detach().cpu().numpy()
            if self.problem == "classification":
                if len(y_pred.shape) == 2:
                    y_pred = np.argmax(y_pred ,axis=1)
                else:
                    y_pred = np.argmax(y_pred.reshape(1, -1), axis=1)
        if self.use_fairness:
            pass
        return get_metric(metric=self.metric, y_pred=y_pred, y=y.values.ravel())

    def compute_chunk_purity(self, fair_attrbs):
        
        if isinstance(fair_attrbs, (pd.DataFrame, pd.Series)):
            values = fair_attrbs.values.flatten()
        elif isinstance(fair_attrbs, np.ndarray):
            values = fair_attrbs.flatten()
        else:
            values = np.asarray(list(fair_attrbs)).flatten()

        if len(values) == 0:
            return 0.0

        _, counts = np.unique(values, return_counts=True)
        return counts.max() / len(values)

    def vanilla_Data_Shapley(self):
        data_dict = self.read_data()

        n = len(data_dict)

        shapley_values = np.zeros(n)
        start_time = datetime.now()
        for i in tqdm(range(n)):
            total_contribution = 0

            for subset_size in range(n):
                subsets = list(itertools.combinations(set(range(n)) - {i}, subset_size))

                for subset in subsets:
                    if len(subset) > 0:
                        subset_with_i = set(subset) | {i}

                        X = [data_['X'] for j, data_ in enumerate(list(data_dict.values())) if j in subset]
                        y = [data_['Y'] for j, data_ in enumerate(list(data_dict.values())) if j in subset]

                        X_with_i = [data_['X'] for j, data_ in enumerate(list(data_dict.values())) if j in subset_with_i]
                        y_with_i = [data_['Y'] for j, data_ in enumerate(list(data_dict.values())) if j in subset_with_i]

                        X = pd.concat(X, axis=0)
                        y = pd.concat(y, axis=0)
                        X_with_i = pd.concat(X_with_i, axis=0)
                        y_with_i = pd.concat(y_with_i, axis=0)
                        weight = math.exp(_shapley_log_p(subset_size, n))
                        value = self.itter_Data_Shapley(X=X, y=y,
                                                        X_with_i=X_with_i, y_with_i=y_with_i,
                                                        weight=weight)

                        total_contribution += value
            shapley_values[i] = total_contribution

        Shap_values_dict = {key: shapley_values[i] for i, key in enumerate(data_dict.keys())}
        self.vanilia_ds = Shap_values_dict
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        return Shap_values_dict


    def itter_Data_Shapley(self, X, y, X_with_i, y_with_i, weight):
        perf_without_i = self.get_score(X=X, y=y, use_img=False, use_NN=False, lr=0.0001, max_epochs=1, batch_size=1, val_frac=0.1)

        perf_with_i = self.get_score(X=X_with_i, y=y_with_i, use_img=False, use_NN=False, lr=0.0001, max_epochs=1, batch_size=1, val_frac=0.1)

        value = weight * (perf_with_i - perf_without_i)
        return value

    def monte_carlo_Data_Shap(self, MC_samples=100):
        data_dict = self.read_data()

        n = len(data_dict)


        start_time = datetime.now()

        shapley_values = np.zeros(n)

        for i in tqdm(range(n)):
            total_contribution = 0
            for _ in range(MC_samples):
                subset = set(random.sample(range(n), random.randint(0, n - 1)))

                if i in subset or len(subset) == 0:
                    continue

                subset_with_i = subset | {i}
                X = [data_['X'] for j, data_ in enumerate(list(data_dict.values())) if j in subset]
                y = [data_['Y'] for j, data_ in enumerate(list(data_dict.values())) if j in subset]

                X_with_i = [data_['X'] for j, data_ in enumerate(list(data_dict.values())) if j in subset_with_i]
                y_with_i = [data_['Y'] for j, data_ in enumerate(list(data_dict.values())) if j in subset_with_i]

                X = pd.concat(X, axis=0)
                y = pd.concat(y, axis=0)
                X_with_i = pd.concat(X_with_i, axis=0)
                y_with_i = pd.concat(y_with_i, axis=0)

                s = len(subset)
                log_q = -math.log(n) - math.lgamma(n + 1) + math.lgamma(s + 1) + math.lgamma(n - s + 1)
                weight = math.exp(_shapley_log_p(s, n) - log_q)
                value = self.itter_Data_Shapley(X=X, y=y,
                                                X_with_i=X_with_i, y_with_i=y_with_i,
                                                weight=weight)

                total_contribution += value
            shapley_values[i] = total_contribution

        Shap_values_dict = {key: shapley_values[i] for i, key in enumerate(data_dict.keys())}
        self.mc_ds = Shap_values_dict
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        return Shap_values_dict

    def execute_per_dataset(self, split_data=False, chunk_size=200, k=50, split_by='ordered', use_hilbert_kernel=False, err=0.001,th=0.5, smart_selection=True, show_subsets=False, data_type='tabular', use_perm=True, ret_embeds=False, convert_label=False, idx_to_label={}, use_fair_permutation=False, method='FLA-SH', r_max=1):

        if self.use_fairness:
            use_fair_permutation = True
        if method == 'C-DaSh' and split_by != 'ordered':
            print(f"[execute_per_dataset] method='C-DaSh' uses order "
                  f"partitioning by definition -- overriding split_by="
                  f"{split_by!r} to 'ordered'.")
            split_by = 'ordered'
        if split_data:
            # Apply chunk partitioning and create chunks
            data, X_embed, fair_attr_groups = self.split_data(chunk_size=chunk_size, split_by=split_by, save_it=True, data_type=data_type, convert_labels=convert_label, idx_to_label=idx_to_label)
            out_avg = {}
            out_shapley_values = {}
            selected_subsets = []
            # for each dataset perform data shapley with his chunks
            for file_name, df_dict in data.items():
                # prepare the chunks for the FLA-SHapley method
                df_dict_prepared = self.prepare_chunks(df_dict=df_dict, class_label=self.label_class)
                if self.use_fairness:
                    input_fair_attrbs = fair_attr_groups[file_name]
                else:
                    input_fair_attrbs = None
                if show_subsets:
                    Shap_values_dict, subsets_ = self.FLA_SHapley(data_dict=df_dict_prepared, df_dict=df_dict, err=err, r_max=r_max, total_subset=k, use_gradient=True, chunk_size=None, th=th, smart_selection=smart_selection, show_subsets=show_subsets, use_hilbert_kernel=use_hilbert_kernel, X_embeds=X_embed, use_perm=use_perm, fair_attr_df=input_fair_attrbs, use_fair_perm=use_fair_permutation, method=method)
                    selected_subsets += subsets_
                else:
                    Shap_values_dict = self.FLA_SHapley(data_dict=df_dict_prepared, df_dict=df_dict, err=err, r_max=r_max, total_subset=k, use_gradient=True, chunk_size=None, th=th, smart_selection=smart_selection, show_subsets=show_subsets, use_hilbert_kernel=use_hilbert_kernel, X_embeds=X_embed, use_perm=use_perm, fair_attr_df=input_fair_attrbs, use_fair_perm=use_fair_permutation, method=method)
                dataset_avg = np.mean(list(Shap_values_dict.values()))
                out_avg[file_name] = dataset_avg
                out_shapley_values[file_name] = Shap_values_dict
            if show_subsets:
                print(f'Selected subsets : ')
                print(selected_subsets)
            if ret_embeds:
                return out_shapley_values, out_avg, X_embed
            self.values_ds_FLA_SH_run = out_shapley_values
            return out_shapley_values, out_avg
        else:
            print('Start FLA-SH Run on pre-divided dataset in chunk')
            if data_type.lower() == 'image' or data_type.lower() == 'img':
                data_dict = self.read_img_data(class_label=self.label_class)
            else:
                data_dict = self.read_data()
            selected_subsets = []
            input_fair_attrbs = None
            if self.use_fairness:
                chunk_dfs = [drop_index_columns(pd.read_csv(self.datasets_filenames[key], index_col=False)).fillna(-99) for key in data_dict.keys()]
                chunk_lengths = [len(df) for df in chunk_dfs]
                fair_attrbs_groups = self.create_group_labels(df_=pd.concat(chunk_dfs, axis=0, ignore_index=True), task_label_col=self.label_class)
                self.num_of_groups = np.unique(fair_attrbs_groups).shape[0]
                split_points = np.cumsum(chunk_lengths)[:-1]
                input_fair_attrbs = list(np.split(fair_attrbs_groups, split_points))
            if show_subsets:
                Shap_values_dict, subsets_ = self.FLA_SHapley(data_dict=data_dict, err=err, r_max=r_max, total_subset=k, use_gradient=True, chunk_size=None,
                                                                       th=th, smart_selection=smart_selection, show_subsets=show_subsets,
                                                                       use_hilbert_kernel=use_hilbert_kernel, X_embeds=None, use_perm=use_perm, fair_attr_df=input_fair_attrbs, method=method)
                selected_subsets += subsets_
            else:
                Shap_values_dict = self.FLA_SHapley(data_dict=data_dict, err=err, r_max=r_max, total_subset=k, use_gradient=True, chunk_size=None, th=th,
                                                             smart_selection=smart_selection, show_subsets=show_subsets, use_hilbert_kernel=use_hilbert_kernel,
                                                             X_embeds=None, use_perm=use_perm, fair_attr_df=input_fair_attrbs, method=method)
            if show_subsets:
                print(f'Selected subsets : ')
                print(selected_subsets)
            dataset_avg = np.mean(list(Shap_values_dict.values()))
            self.values_ds_FLA_SH_run = Shap_values_dict
            if ret_embeds:
                return out_shapley_values, out_avg, X_embed
            return Shap_values_dict, dataset_avg

    def subset_selection(self, data_dict, data_X, data_y, data_fair, subsets, lr, total_subset, th,
                         metrics, data_type, show_subsets=False, val_frac=0.1, use_gradient=True,
                         use_norm=True, fair_attrbs=None, exec_per_point=False):
        data_values = list(data_dict.values())
        data_dict_keys_lst = list(data_dict.keys())
        data_flags = {key: True for key in data_dict_keys_lst}
        data_count = {key: 0 for key in data_dict_keys_lst}

        allowed_perc = np.ceil(total_subset * 0.25)
        priority_counter = 5

        # Pre-define comparator
        if self.problem == 'regression':
            eq_ = lambda a, b: a > b 
        else:
            eq_ = lambda a, b: a < b 

        for idx in tqdm(range(len(subsets))):

            flag = False
            for chunk_id in subsets[idx]:
                # Using the pre-computed keys list
                if data_count[data_dict_keys_lst[chunk_id]] >= allowed_perc:
                    flag = True
                    break 

            attempts = 0
            
            while (eq_(metrics[idx], th) or flag):
                if attempts >= 2:
                    break

                attempts += 1
                
                
                range_limit = idx - 2 if idx == len(data_dict_keys_lst) else idx + 2
                range_limit = max(1, range_limit) 

                sample_size = random.randint(0, range_limit)
                new_subset = list(set(random.sample(range(range_limit), sample_size)))
                
                # Retry logic if empty
                while len(new_subset) == 0:
                    sample_size = random.randint(0, range_limit)
                    new_subset = list(set(random.sample(range(range_limit), sample_size)))
                
                
                valid_candidates = [i for i, k in enumerate(data_dict_keys_lst) if data_count[k] < allowed_perc]
                valid_candidates_len = len(valid_candidates)

                if valid_candidates_len <= 1:
                    pass 
                else:
                    for j, chunk_id in enumerate(new_subset):
                        data_key = data_dict_keys_lst[chunk_id]
                        
                        if data_count[data_key] >= allowed_perc:
                            if valid_candidates_len == 0:
                                new_subset[j] = random.randint(0, idx + priority_counter)
                            else:
                                new_subset[j] = random.choice(valid_candidates)

                flag = False
                new_subset = set(new_subset)
                subsets[idx] = new_subset

                X = [data_values[j]['X'] for j in new_subset]
                y = [data_values[j]['Y'] for j in new_subset]

                if self.use_fairness:

                    fair_attrb_ = [fair_attrbs[j] for j in new_subset]

                if use_gradient:
                    if all(isinstance(item, pd.DataFrame) for item in X):
                        X = pd.concat(X, axis=0)
                        y = pd.concat(y, axis=0)
                    elif all(isinstance(item, np.ndarray) for item in X):
                        X = np.concatenate(X, axis=0)
                        y = np.concatenate(y, axis=0)
                                        
                    if self.use_fairness:
                        if all(isinstance(item, (pd.DataFrame, pd.Series)) for item in fair_attrb_):
                            fair_attrb_ = pd.concat(fair_attrb_, axis=0)
                        elif all(isinstance(item, np.ndarray) for item in fair_attrb_):
                            fair_attrb_ = np.concatenate(fair_attrb_, axis=0)
                        
                        data_fair[idx] = fair_attrb_
                    
                    data_X[idx] = X
                    data_y[idx] = y
                    
                    current_len = len(X)
                    if current_len * val_frac < 2:
                        required_frac = 2.0 / current_len
                        val_frac = np.ceil(required_frac * 100) / 100
                    
                    if self.use_embeds and self.use_img:
                        model = create_model(model_type=self.model_type, 
                                        problem=self.problem,
                                        max_epochs=1,
                                        batch_size=1, 
                                        lr=lr, 
                                        val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=False, 
                                        use_NN=True,
                                        num_classes=self.num_of_classes, 
                                        norm=use_norm,
                                        use_fairness=self.use_fairness,
                                        fairness_attr=self.fairness_attr,
                                        num_of_fair=self.num_of_groups, 
                                        lambda_fair=self.lambda_fair,
                                        fair_metric=self.fair_metric)

                    else:
                        model = create_model(model_type=self.model_type, 
                                        problem=self.problem,
                                        max_epochs=1,
                                        batch_size=1, 
                                        lr=lr, 
                                        val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=self.use_img, 
                                        use_NN=True,
                                        num_classes=self.num_of_classes, 
                                        norm=use_norm,
                                        use_fairness=self.use_fairness,
                                        fairness_attr=self.fairness_attr,
                                        num_of_fair=self.num_of_groups, 
                                        lambda_fair=self.lambda_fair,
                                        fair_metric=self.fair_metric)
                    model = self._configure_model(model)

                    fair_attrbs_train = fair_attrb_ if self.use_fairness else None
                    fair_attrbs_val = None
                    if exec_per_point:
                        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
                        if n_samples >= 10:
                            try:
                                if self.use_fairness:
                                    X_train, X_val, y_train, y_val, fair_attrbs_train, fair_attrbs_val = train_test_split(
                                        X, y, fair_attrb_, test_size=0.1, random_state=42)
                                else:
                                    X_train, X_val, y_train, y_val = train_test_split(
                                        X, y, test_size=0.1, random_state=42)
                            except Exception:
                                X_train, y_train, X_val, y_val = X, y, X, y
                                fair_attrbs_val = fair_attrb_ if self.use_fairness else None
                        else:
                            X_train, y_train, X_val, y_val = X, y, X, y
                            fair_attrbs_val = fair_attrb_ if self.use_fairness else None
                    else:
                        X_train = X
                        y_train = y
                        X_val = None
                        y_val = None
                    model.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val, fair_attrbs=fair_attrbs_train, fair_attrbs_val=fair_attrbs_val, data_dict=None, batch_size=1, epochs=1, metric=self.metric)
                    if model.history['metrics'] and idx < len(model.history['metrics']):
                        new_metric = model.history['metrics'][idx]
                    elif model.history['metrics']:
                        new_metric = model.history['metrics'][-1]
                    else:
                        new_metric = 0.0 
                else:
                    if self.use_img and self.use_embeds:
                        new_metric = self.get_score(X=X, y=y, use_img=False, use_NN=False, lr=lr, max_epochs=1, batch_size=1, val_frac=0.1, data_type=data_type)
                    else:
                        new_metric = self.get_score(X=X, y=y, use_img=self.use_img, use_NN=False, lr=lr, max_epochs=1, batch_size=1, val_frac=0.1, data_type=data_type)
                
                if eq_(metrics[idx], new_metric):
                    subsets[idx] = new_subset
                    metrics[idx] = new_metric

            for chunk_id in subsets[idx]:
                key = data_dict_keys_lst[chunk_id]
                data_count[key] += 1
                if data_count[key] > allowed_perc:
                    data_flags[key] = False
                    priority_counter += 1
                    if priority_counter >= len(data_count):
                        priority_counter = len(data_count) - 1
            
        return subsets, metrics
    
    def _fit_feature_transform(self, data_dict):
        self.feature_transform = None
        if self.preprocessing != 'onehot_zscore' or self.use_img or self.use_embeds:
            return
        X_parts = [v['X'] for v in data_dict.values()]
        if not all(isinstance(x, pd.DataFrame) for x in X_parts):
            return
        self.feature_transform = FeatureTransform(pd.concat(X_parts, axis=0))

    def _configure_model(self, model):
        if getattr(self, 'feature_transform', None) is not None:
            model.feature_transform = self.feature_transform
            model.normalize = False
        model.utility = getattr(self, 'utility', 'accuracy')
        if getattr(self, 'split_rng', None) is not None:
            model.split_rng = self.split_rng
        if getattr(self, 'utility_source', 'coalition_split') != 'fixed_val':
            return model
        if self.use_fairness or self.use_img:
            raise NotImplementedError("utility_source='fixed_val' supports tabular, non-fairness runs only")
        from sklearn import preprocessing
        X_val, y_val = self.fixed_val_xy
        if getattr(model, 'feature_transform', None) is not None:
            X_val = model.feature_transform(X_val)
        else:
            X_val = np.asarray(X_val, dtype=np.float64)
            if model.normalize:
                X_val = preprocessing.normalize(X_val)   # same per-row L2 norm training_ applies
        model.fixed_val_data = (transform_to_torch(X_val).to(model.device),
                                transform_to_torch(np.asarray(y_val).ravel()).to(model.device).ravel())
        return model

    def FLA_SHapley(self, data_dict=None, df_dict=None, err=0.001, total_subset=50, use_gradient=False, lr=0.001, chunk_size=None,
                    th=0.6, smart_selection=True, val_frac=0.1, show_subsets=False, use_hilbert_kernel=False, use_perm=True,
                    SR=0.1, X_embeds=None, data_type='tabular', load_embeds=False, fair_attr_df=False,
                    use_fair_perm=False, alpha=0.7, exec_per_point=False,
                    epsilon=0.1, priority_strength=0.5, method='FLA-SH', r_max=1):
        """
        `method` selects which coalition-sampling / weighting scheme is used: 'FLA-SH' (default) or 'C-DaSh'
        Round control
                  err   : truncation tolerance EPSILON on THIS ROUND's mean absolute
                          marginal contribution.
                  r_max : hard cap on the number of rounds (R_max). The round loop
                          runs at most r_max rounds regardless of whether the
                          truncation criterion above is ever met.
         """
        if method not in ('FLA-SH', 'C-DaSh'):
            raise ValueError(f"FLA_SHapley: unknown method={method!r}, expected 'FLA-SH' or 'C-DaSh'")
        if method == 'FLA-SH' and smart_selection:
            print('[FLA_SHapley]: Run original method')
        elif method == 'C-DaSh':
            print('[FLA_SHapley] but using the C-DaSh approximation method')

        if data_type == 'text':
            use_norm = False
        else:
            use_norm = True

        if data_dict is None:
            data, X_embeds, fair_attr_df = self.split_data(chunk_size=chunk_size, split_by='', save_it=True, data_type=data_type, convert_labels=False)
            data_dict = self.prepare_chunks(df_dict=data, class_label=self.label_class)
        elif not isinstance(data_dict, dict):
            data_dict = self.read_data()

        self._fit_feature_transform(data_dict)

        n = len(data_dict)
        if n <= 1:
            Shap_values_dict = {key: 0 for i, key in enumerate(data_dict.keys())}
            self.values_ds_FLA_SH_run = Shap_values_dict
            return Shap_values_dict
        
        if n < total_subset:
            print(f'Change subset size to the length of the chunks to {n/2}, because cannot be the chunk size k > n')
            total_subset = int(n/2)

        if self.problem == 'classification':
            all_labels = set()
            for _vals in data_dict.values():
                _y = _vals['Y']
                if isinstance(_y, pd.DataFrame):
                    all_labels.update(_y.values.flatten().tolist())
                elif isinstance(_y, pd.Series):
                    all_labels.update(_y.tolist())
                else:
                    all_labels.update(np.array(_y).ravel().tolist())

            try:
                int_labels = [int(l) for l in all_labels]
                required = max(int_labels) + 1
            except (ValueError, TypeError):
                required = len(all_labels)

            required = max(2, required)

            if self.num_of_classes is None or self.num_of_classes < required:
                print(f'FLA_SHapley: adjusting num_of_classes {self.num_of_classes} -> {required} (max label={max(int_labels) if "int_labels" in dir() else "?"}, unique={len(all_labels)})')
                self.num_of_classes = required

        _all_y_pooled = []
        for _vals in data_dict.values():
            _y = _vals['Y']
            if isinstance(_y, pd.DataFrame):
                _all_y_pooled.append(_y.values.ravel())
            elif isinstance(_y, pd.Series):
                _all_y_pooled.append(_y.to_numpy().ravel())
            else:
                _all_y_pooled.append(np.asarray(_y).ravel())
        _all_y_pooled = np.concatenate(_all_y_pooled)

        _metric_l = self.metric.lower()
        if getattr(self, 'utility', 'accuracy') == 'neg_val_ce':
            _vals_, _counts_ = np.unique(_all_y_pooled, return_counts=True)
            _p_ = _counts_ / _counts_.sum()
            v_empty = float(np.sum(_p_ * np.log(_p_)))
        elif _metric_l == 'accuracy':
            _vals_, _counts_ = np.unique(_all_y_pooled, return_counts=True)
            v_empty = float(_counts_.max() / _counts_.sum())
        elif _metric_l == 'f1-score':
            _vals_, _counts_ = np.unique(_all_y_pooled, return_counts=True)
            _maj_label = _vals_[np.argmax(_counts_)]
            _const_pred = np.full_like(_all_y_pooled, _maj_label)
            v_empty = float(get_metric(metric='f1-score', y_pred=_const_pred, y=_all_y_pooled))
        elif _metric_l in ('mse', 'rmse'):
            _mse_ = float(np.var(_all_y_pooled))
            v_empty = _mse_ if _metric_l == 'mse' else float(np.sqrt(_mse_))
        elif _metric_l == 'auc':
            v_empty = 0.5
        else:
            v_empty = 0.0

        # Perform Hilbert Kernel Permutation
        if use_hilbert_kernel and use_perm:
            print('Start Hilbert permutation')
            
            hilbert_bits = 10
            dim = 2
            priorities = np.random.rand(n)

            # For data types that embeddings required check if exists otherwise re-compute it
            if self.use_img:
                if X_embeds is None:
                    data_dim = []
                    i = 0
                    start_time = time.time()
                    for filename, chunk in df_dict.items():
                        
                        target_col = None
                        for col in ['file_path', 'file', 'filename']:
                            if col in chunk.columns:
                                target_col = col
                                break
                        Y_labels = {row[target_col]: row[self.label_class] for index, row in chunk.iterrows()}

                        if self.use_fairness:
                            fair_feat = fair_attr_df[i]
                            i += 1
                        else:
                            fair_feat = None
                        embeds, img_embeds = self.convert_img_data_to_vectors(data=chunk,  y_labels=Y_labels, fair_feat=fair_feat)
                        data_dim.append(np.mean(list(embeds.values()), axis=0))
                        
                    data_dim = np.asarray(data_dim)
                    dim = data_dim[0].shape[0]
                    print(f'Finish embedding image data in {time.time() - start_time:.2f} seconds')
                else:
                    dim = X_embeds[0].shape[1]
                    
                    data_dim = []
                    for arr in X_embeds:
                        data_dim.append(np.mean(arr, axis=0))
            elif data_type.lower() == 'text' or data_type.lower() == 'textual':
                data_dim = []
                if X_embeds is None:
                    X_embeds_lst = []
                    if load_embeds:
                        for filename, chunk in df_dict.items():
                            X_embds = chunk['X']
                            data_dim.append(np.mean(list(X_embds), axis=0))
                            X_embeds_lst.append(X_embds)   # FIX: was X_embeds (outer param)
                        data_dim = np.asarray(data_dim)
                        dim = data_dim[0].shape[0]
                        X_embeds = X_embeds_lst 
                    else:
                        start_time = time.time()
                        for filename, chunk in df_dict.items():
                            X_embds, df_text_info = self.convert_text_data_to_vectors(data=chunk)
                            data_dim.append(np.mean(list(X_embds.values()), axis=0))
                            X_embeds_lst.append(np.array(list(X_embds.values())))  # FIX: was X_embeds
                        data_dim = np.asarray(data_dim)
                        dim = data_dim[0].shape[0]
                        X_embeds = X_embeds_lst
                        print(f'Finish embedding text data in {time.time() - start_time:.2f} seconds')
                else:
                    data_dim = []
                    for arr in X_embeds:
                        arr = np.array(arr)
                        if arr.ndim == 1:
                            data_dim.append(arr)
                            dim = arr.shape[0]
                        else:
                            data_dim.append(np.mean(arr, axis=0))
                            dim = arr.shape[1]
            
            elif data_type.lower() == 'tabular':
                start_time = time.time()
                if exec_per_point:
                    data_arrays = [
                        pd.concat([data_['X'], data_['Y']], axis=1).to_numpy().flatten()
                        for name, data_ in data_dict.items()
                    ]
                    max_dim = max(len(a) for a in data_arrays)
                    if max_dim >= 2:
                        all_data = np.vstack([a.reshape(1, -1) for a in data_arrays])
                        pca = PCA(n_components=min(2, all_data.shape[0], all_data.shape[1]))
                        all_embedded = pca.fit_transform(all_data)
                        data_dim = [all_embedded[i] for i in range(len(data_arrays))]
                        dim = all_embedded.shape[1]
                    else:
                        data_dim = [np.pad(a, (0, 2 - len(a))) for a in data_arrays]
                        dim = 2
                else:
                    pca = PCA(n_components=dim)
                    data_dim = [
                        pd.concat([data_['X'], data_['Y']], axis=1).to_numpy()
                        for name, data_ in data_dict.items()
                    ]
                    data_dim = [np.mean(pca.fit_transform(arr), axis=0) for arr in data_dim]
                print(f'Finish embedding tabular data with PCA in {time.time() - start_time:.2f} seconds')

            data_names = [name for name, data_ in data_dict.items()]
            data_dim = np.vstack(data_dim)

            start_time = time.time()
            scaled_positions = scale_to_grid(data_dim, hilbert_bits)

            hilbert_curve = HilbertCurve(hilbert_bits, dim)
            
            hilbert_indices = np.array([hilbert_curve.distance_from_point(coord.tolist()) for coord in scaled_positions])

            original_indices = range(len(data_names))

            objects = list(zip(data_names, data_dim, priorities, hilbert_indices, original_indices))

            sorted_chunked = sorted(objects, key=lambda x: x[3], reverse=False)
            
            data_dict_tmp = {item[0]: data_dict[item[0]] for item in sorted_chunked}
            data_dict = data_dict_tmp
            
            if self.use_fairness:
                fair_attr_df = [fair_attr_df[item[4]] for item in sorted_chunked]
            print(f'Finish Hilbert Permutation in {time.time() - start_time:.2f} seconds')
            print('Finish Hilbert Permutation')

        elif use_fair_perm and self.use_fairness and use_perm:
            print(f'Start fairness-aware purity permutation (alpha={alpha})')
            start_time = time.time()

            purity_scores = {}
            key_to_original_idx = {}
            for i, (key, _) in enumerate(data_dict.items()):
                key_to_original_idx[key] = i
                purity_scores[key] = self.compute_chunk_purity(fair_attr_df[i])

            eligible_keys = [k for k, p in purity_scores.items() if p >= alpha]

            if len(eligible_keys) < 2:
                print(f'Warning: only {len(eligible_keys)} chunks satisfy alpha={alpha}. '
                        f'Falling back to using all chunks sorted by purity.')
                eligible_keys = list(purity_scores.keys())

            eligible_keys = sorted(eligible_keys, key=lambda k: purity_scores[k], reverse=True)

            data_dict = {key: data_dict[key] for key in eligible_keys}
            if self.use_fairness:
                fair_attr_df = [fair_attr_df[key_to_original_idx[key]] for key in eligible_keys]

            print(f'Selected {len(eligible_keys)} chunks above alpha={alpha}')
            print(f'Finish fairness-aware permutation in {time.time() - start_time:.2f} seconds')

        elif use_perm:
            print('Start SR permutation')
            start_time = time.time()
            scores_tmp = {key: 0 for key, values in data_dict.items()}
            key_to_original_idx = {}
            for i, (key, values) in enumerate(data_dict.items()):
                key_to_original_idx[key] = i
                random_indices = random.sample(range(values['X'].shape[0]), np.ceil(values['X'].shape[0]*SR).astype(np.int_))
                
                if isinstance(values['X'], pd.DataFrame):
                    X_ = values['X'].iloc[random_indices]
                else:
                    X_ = values['X'][random_indices]
                
                if isinstance(values['Y'], pd.DataFrame):
                    y_ = values['Y'].iloc[random_indices]
                else:
                    y_ = values['Y'][random_indices]

                sensitive_fair_attrbs = None
                if self.use_fairness:
                    fair_attrbs = fair_attr_df[i]
                    if isinstance(fair_attrbs, (pd.DataFrame, pd.Series)):
                        sensitive_fair_attrbs = fair_attrbs.iloc[random_indices]
                    elif isinstance(fair_attrbs, np.ndarray):
                        sensitive_fair_attrbs = fair_attrbs[random_indices]
                    else:
                        sensitive_fair_attrbs = [fair_attrbs[i] for i in random_indices]

                if self.use_embeds and self.use_img:
                    score_ = self.get_score(X=X_, y=y_, use_img=False, use_NN=True, lr=lr, max_epochs=1, batch_size=1, val_frac=val_frac, data_type=data_type, fair_attr_df=sensitive_fair_attrbs, ) 
                else:
                    score_ = self.get_score(X=X_, y=y_, use_img=self.use_img, use_NN=True, lr=lr, max_epochs=1, batch_size=1, val_frac=val_frac, data_type=data_type, fair_attr_df=sensitive_fair_attrbs) 
                scores_tmp[key] = score_

            scores_tmp = dict(sorted(scores_tmp.items(), key=lambda item: item[1], reverse=True))
            sorted_keys = list(scores_tmp.keys())
            data_dict = {key: data_dict[key] for key in sorted_keys}
            if self.use_fairness:
                fair_attr_df = [fair_attr_df[key_to_original_idx[key]] for key in sorted_keys]
            print(f'Finish SR permutation in {time.time() - start_time:.2f} seconds')
        
        memory_ds = np.zeros((n, total_subset))

        if method != 'C-DaSh':
            _pi = _chunk_inclusion_probs(n, strength=priority_strength)
            _coalition_seed = np.random.randint(0, 2**31 - 1)
            if getattr(self, 'coalition_seed', None) is not None:
                _coalition_seed = self.coalition_seed
            _coalition_rng = np.random.default_rng(_coalition_seed)

        assert r_max >= 1, f"r_max must be >= 1, got {r_max}"
        round_mean_abs_delta = None
        self.last_fla_sh_order = list(data_dict.keys())
        self.last_fla_sh_subsets = []
        self.last_fla_sh_deltas = []
        self.last_fla_sh_pairs = []
        for itter_count in range(r_max):
            metrics = []
            subsets = []
            importance_weights = None
            if method == 'C-DaSh':
                for _ in range(total_subset):
                    subset = set()
                    attempts = 0
                    while len(subset) == 0 and attempts < 20:
                        size = random.randint(0, n - 1)
                        subset = set(random.sample(range(n), size))
                        attempts += 1
                    if len(subset) == 0:
                        subset = {random.randrange(n)}
                    subsets.append(subset)
            else:
                
                importance_weights = []
                for _ in range(total_subset):
                    _Z, _w = _sample_coalition_qtilde(n, _pi, epsilon, _coalition_rng)
                    subsets.append(_Z)
                    importance_weights.append(_w)
            print(f'Itteration {itter_count+1}/{r_max}')
            self.last_fla_sh_subsets.append([sorted(s) for s in subsets])

            if use_gradient:
                data_X = []
                data_y = []
                data_fair = []
            included_original_idxs = []

            for idx, subset in enumerate(tqdm(subsets)):
                if len(subset) == 0:
                    metrics.append(v_empty)
                    continue

                X = [data_['X'] for j, data_ in enumerate(list(data_dict.values())) if j in subset]
                y = [data_['Y'] for j, data_ in enumerate(list(data_dict.values())) if j in subset]

                if self.use_fairness:
                    fair_attrb_ = [data_ for j, data_ in enumerate(fair_attr_df) if j in subset]

                if all(isinstance(item, pd.DataFrame) for item in X):
                    X = pd.concat(X, axis=0)
                    y = pd.concat(y, axis=0)
                elif all(isinstance(item, np.ndarray) for item in X):
                    X = np.concatenate(X, axis=0)
                    y = np.concatenate(y, axis=0)

                if self.use_fairness:
                    if all(isinstance(item, (pd.DataFrame, pd.Series)) for item in fair_attrb_):
                        fair_attrb_ = pd.concat(fair_attrb_, axis=0)
                    elif all(isinstance(item, np.ndarray) for item in fair_attrb_):
                        fair_attrb_ = np.concatenate(fair_attrb_, axis=0)

                if use_gradient:
                    data_X.append(X)
                    data_y.append(y)
                    if self.use_fairness:
                        data_fair.append(fair_attrb_)
                    included_original_idxs.append(idx)
                    metrics.append(0)
                else:
                    if self.use_embeds and self.use_img:
                        metrics.append(self.get_score(X=X_, y=y_, use_img=False, use_NN=False, lr=lr, max_epochs=1, batch_size=1, val_frac=0.1, data_type=data_type))
                    else:
                        metrics.append(self.get_score(X=X_, y=y_, use_img=self.use_img, use_NN=False, lr=lr, max_epochs=1, batch_size=1, val_frac=0.1, data_type=data_type))

            if use_gradient and data_X:
                if self.use_embeds and self.use_img:
                    model = create_model(model_type=self.model_type, 
                                        problem=self.problem, max_epochs=1, batch_size=1, lr=lr, 
                                        val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=False, 
                                        use_NN=True, num_classes=self.num_of_classes, norm=use_norm,
                                        use_fairness=self.use_fairness, fairness_attr=self.fairness_attr,
                                        num_of_fair=self.num_of_groups, lambda_fair=self.lambda_fair,
                                        fair_metric=self.fair_metric)
                else:
                    model = create_model(model_type=self.model_type, 
                                        problem=self.problem, max_epochs=1, batch_size=1, lr=lr, 
                                        val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=self.use_img, 
                                        use_NN=True, num_classes=self.num_of_classes, norm=use_norm,
                                        use_fairness=self.use_fairness, fairness_attr=self.fairness_attr,
                                        num_of_fair=self.num_of_groups, lambda_fair=self.lambda_fair,
                                        fair_metric=self.fair_metric)
                model = self._configure_model(model)

                fair_attrbs_train = data_fair if self.use_fairness else None
                model.fit(X=data_X, y=data_y, X_val=None, y_val=None, fair_attrbs=fair_attrbs_train, data_dict=None, batch_size=1, epochs=1, metric=self.metric,
                          release_inputs=(method != 'C-DaSh'))

                for i, index in enumerate(model.history['idxs']):
                    metrics[included_original_idxs[index]] = model.history['metrics'][i]

            if method == 'C-DaSh':
                if use_gradient:
                    subsets, metrics = self.subset_selection(
                        data_dict=data_dict, data_X=data_X, data_y=data_y, data_fair=data_fair,
                        subsets=subsets, lr=lr, total_subset=total_subset, th=th, metrics=metrics,
                        data_type=data_type, show_subsets=show_subsets, val_frac=val_frac,
                        use_gradient=use_gradient, use_norm=use_norm, fair_attrbs=fair_attr_df,
                        exec_per_point=exec_per_point)
                else:
                    print("[FLA_SHapley] method='C-DaSh' with use_gradient=False: "
                          "subset_selection() only implements the gradient-batched path -- "
                          "skipping the resample filter for this call (using the initial "
                          "uniform-random subsets/metrics unfiltered).")
            else:
                pass

            if show_subsets:
                subset_names = []
                for subset in subsets:
                    for k, data_name in enumerate(list(data_dict.keys())):
                        if k in subset:
                            subset_names.append(data_name)
            
            marginal_contrib = np.zeros((n, total_subset))
            raw_delta = np.zeros((n, total_subset))   # unweighted M(Z u {i}) - M(Z), for Eq. 4
            computed_mask = np.zeros((n, total_subset), dtype=bool)   

            if self.use_embeds and self.use_img:
                model = create_model(model_type=self.model_type,
                                            problem=self.problem, max_epochs=1, batch_size=1, lr=lr,
                                            val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=False,
                                            use_NN=True, num_classes=self.num_of_classes, norm=use_norm,
                                            use_fairness=self.use_fairness, fairness_attr=self.fairness_attr,
                                            num_of_fair=self.num_of_groups, lambda_fair=self.lambda_fair,
                                            fair_metric=self.fair_metric)
            else:
                model = create_model(model_type=self.model_type, 
                                            problem=self.problem, max_epochs=1, batch_size=1, lr=lr, 
                                            val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=self.use_img, 
                                            use_NN=True, num_classes=self.num_of_classes, norm=use_norm,
                                            use_fairness=self.use_fairness, fairness_attr=self.fairness_attr,
                                            num_of_fair=self.num_of_groups, lambda_fair=self.lambda_fair,
                                            fair_metric=self.fair_metric)
            model = self._configure_model(model)

            for i in tqdm(range(n)):
                for j, subset in enumerate(subsets):
                    if i in subset:
                        continue
                    subset_with_i = subset | {i}
                    X = [data_['X'] for k, data_ in enumerate(list(data_dict.values())) if k in subset_with_i]
                    y = [data_['Y'] for k, data_ in enumerate(list(data_dict.values())) if k in subset_with_i]
                    if self.use_fairness:
                        fair_attrb_ = [data_ for k, data_ in enumerate(fair_attr_df) if k in subset_with_i]
                        if all(isinstance(item, (pd.DataFrame, pd.Series)) for item in fair_attrb_):
                            fair_attrb_ = pd.concat(fair_attrb_, axis=0)
                        elif all(isinstance(item, np.ndarray) for item in fair_attrb_):
                            fair_attrb_ = np.concatenate(fair_attrb_, axis=0)
                    else:
                        fair_attrb_ = None
                    
                    if all(isinstance(item, pd.DataFrame) for item in X):
                        X = pd.concat(X, axis=0)
                        y = pd.concat(y, axis=0)
                    elif all(isinstance(item, np.ndarray) for item in X):
                        X = np.concatenate(X, axis=0)
                        y = np.concatenate(y, axis=0)
                    
                    if use_gradient:
                        fair_attrbs_train = fair_attrb_ if self.use_fairness else None
                        fair_attrbs_val = None
                        if exec_per_point:
                            n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
                            if n_samples >= 10:
                                try:
                                    if self.use_fairness:
                                        X_train, X_val, y_train, y_val, fair_attrbs_train, fair_attrbs_val = train_test_split(
                                            X, y, fair_attrb_, test_size=0.1, random_state=42)
                                    else:
                                        X_train, X_val, y_train, y_val = train_test_split(
                                            X, y, test_size=0.1, random_state=42)
                                except Exception:
                                    X_train, y_train, X_val, y_val = X, y, X, y
                                    fair_attrbs_val = fair_attrb_ if self.use_fairness else None
                            else:
                                X_train, y_train, X_val, y_val = X, y, X, y
                                fair_attrbs_val = fair_attrb_ if self.use_fairness else None
                        else:
                            X_train = X
                            y_train = y
                            X_val = None
                            y_val = None
                        model.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val, fair_attrbs=fair_attrbs_train, fair_attrbs_val=fair_attrbs_val, data_dict=None, batch_size=1, epochs=1, metric=self.metric)
                        perf_with_i = model.history['metrics'][-1]
                    else:
                        if self.use_img and self.use_embeds:
                            perf_with_i = self.get_score(X=X_, y=y_, use_img=False, use_NN=False, lr=lr, max_epochs=1, batch_size=1, val_frac=0.1, data_type=data_type)
                        else:
                            perf_with_i = self.get_score(X=X_, y=y_, use_img=self.use_img, use_NN=False, lr=lr, max_epochs=1, batch_size=1, val_frac=0.1, data_type=data_type)

                    if method == 'C-DaSh':
                        weight = 1.0 / (n * (len(subset) + 1))
                    else:
                        weight = importance_weights[j]
                    self.last_fla_sh_deltas.append(perf_with_i - metrics[j])
                    self.last_fla_sh_pairs.append((i, perf_with_i, metrics[j]))
                    value = weight * (perf_with_i - metrics[j])
                    marginal_contrib[i, j] = value
                    raw_delta[i, j] = perf_with_i - metrics[j]
                    computed_mask[i, j] = True

            if itter_count == 0:
                memory_ds = marginal_contrib
            else:
                memory_ds = np.concatenate([memory_ds, marginal_contrib])

            if computed_mask.any():
                round_mean_abs_delta = float(np.abs(raw_delta[computed_mask]).mean())
            else:
                round_mean_abs_delta = 0.0
            print(f'Round {itter_count+1}/{r_max}: mean |marginal contribution| '
                  f'(Eq. 4) = {round_mean_abs_delta:.6f}  (epsilon={err})')
            if itter_count + 1 >= 2 and round_mean_abs_delta < err:
                print(f'Truncation criterion met after {itter_count+1}/{r_max} round(s) '
                      f'-- stopping.')
                break
        else:
            print(f'Round budget R_max={r_max} reached without meeting the truncation '
                  f'criterion (last round\'s mean |marginal contribution| = '
                  f'{round_mean_abs_delta:.6f}, epsilon={err}) -- stopping.')

        Shap_values_dict = {key: np.mean(memory_ds[i]) for i, key in enumerate(data_dict.keys())}
        self.values_ds_FLA_SH_run = Shap_values_dict

        self.last_fla_sh_rounds_run = itter_count + 1

        if show_subsets:
            return Shap_values_dict, subset_names
        else:
            if exec_per_point:
                return Shap_values_dict, data_dict
            else:
                return Shap_values_dict

    def FLA_SHapley_per_point(self, data_dict=None, df_dict=None, err=0.001, total_subset=50,
                          use_gradient=True, lr=0.001, th=0.6, smart_selection=True,
                          val_frac=0.1, show_subsets=False, use_hilbert_kernel=False,
                          use_perm=True, SR=0.1, X_embeds=None, data_type='tabular',
                          fair_attr_df=None, use_fair_perm=False, alpha=0.7,
                          max_points=None):
        use_norm = (data_type != 'text')
 
        def _normalise_chunk(key, val):
            
            if isinstance(val, dict) and 'X' in val and 'Y' in val:
                return val

            if isinstance(val, pd.DataFrame):
                class_cols = [c for c in val.columns if c.startswith(self.label_class)]
                embed_cols = [c for c in val.columns if c.startswith('X_') and c[2:].isdigit()]
                if embed_cols:
                    x_df = val[embed_cols]
                else:
                    drop_cols = list(class_cols)
                    for fc in ['file_path', 'filepath', 'file', 'filename']:
                        if fc in val.columns:
                            drop_cols.append(fc)
                            break
                    x_df = val.drop(columns=drop_cols, errors='ignore')
                y_df = val[class_cols] if class_cols else val[[val.columns[-1]]]
                return {'X': x_df, 'Y': y_df}

            if isinstance(val, dict) and all(isinstance(v, pd.DataFrame) for v in val.values()):
                combined = pd.concat(list(val.values()), axis=0).reset_index(drop=True)
                return _normalise_chunk(key, combined)  

            raise ValueError(
                f"Cannot normalise chunk '{key}': got {type(val)} with keys: {list(val.keys()) if isinstance(val, dict) else 'N/A'}")
 
        if data_dict is None:
            out_dict, X_embeds, fair_attr_df = self.split_data( chunk_size=1, split_by='ordered', save_it=False, data_type=data_type, convert_labels=False)

            data_dict = {}
            for _fn, _inner in out_dict.items():
                for _cname, _cval in _inner.items():
                    _key = _cname.split('/')[-1].split('.')[0].upper()
                    data_dict[_key] = _normalise_chunk(_key, _cval)
        else:
            
            data_dict = {k: _normalise_chunk(k, v)
                         for k, v in data_dict.items()}
 
        point_dict = {}
        point_fair = [] if self.use_fairness else None
        point_embeds = [] if X_embeds is not None else None
 
        for c_idx, (chunk_key, chunk_val) in enumerate(data_dict.items()):
            X_c = chunk_val['X']
            y_c = chunk_val['Y']
            n_c = len(X_c)
 
            fair_c = fair_attr_df[c_idx] if self.use_fairness else None
            emb_c = X_embeds[c_idx] if point_embeds is not None else None
 
            for r in range(n_c):
                pkey = f'{chunk_key}__pt{r}'
 
                if isinstance(X_c, pd.DataFrame):
                    X_row = X_c.iloc[[r]]
                    y_row = (y_c.iloc[[r]] if isinstance(y_c, pd.DataFrame)
                             else y_c[[r]])
                else:
                    X_row = X_c[r:r+1]
                    y_row = (y_c.iloc[[r]] if isinstance(y_c, pd.DataFrame)
                             else y_c[r:r+1])
 
                point_dict[pkey] = {'X': X_row, 'Y': y_row}
 
                if self.use_fairness:
                    if isinstance(fair_c, (pd.DataFrame, pd.Series)):
                        point_fair.append(fair_c.iloc[[r]])
                    elif isinstance(fair_c, np.ndarray):
                        point_fair.append(fair_c[r:r+1])
                    else:
                        point_fair.append(np.array([fair_c[r]]))
 
                if point_embeds is not None:
                    point_embeds.append(emb_c[r:r+1])
 
        if max_points is not None and len(point_dict) > max_points:
            keep_keys = random.sample(list(point_dict.keys()), max_points)
            kept_idx = [i for i, k in enumerate(point_dict.keys())
                         if k in set(keep_keys)]
            point_dict = {k: point_dict[k] for k in keep_keys}
            if self.use_fairness:
                point_fair = [point_fair[i] for i in kept_idx]
            if point_embeds is not None:
                point_embeds = [point_embeds[i] for i in kept_idx]
 
        n = len(point_dict)
        if n <= 1:
            zeros = {k: 0.0 for k in point_dict}
            self.values_ds_FLA_SH_run = zeros
            return zeros
 
        if n < total_subset:
            total_subset = max(2, int(n / 2))
 
        all_labels = set()
        for val in point_dict.values():
            y_chunk = val['Y']
            if isinstance(y_chunk, pd.DataFrame):
                all_labels.update(y_chunk.values.flatten().tolist())
            elif isinstance(y_chunk, pd.Series):
                all_labels.update(y_chunk.tolist())
            else:
                all_labels.update(np.array(y_chunk).ravel().tolist())
        self.num_of_classes = max(2, len(all_labels))
 
        return self.FLA_SHapley(
            data_dict=point_dict,
            df_dict=None,
            err=err,
            total_subset=total_subset,
            use_gradient=use_gradient,
            lr=lr,
            chunk_size=None,
            th=th,
            smart_selection=smart_selection,
            val_frac=val_frac,
            show_subsets=show_subsets,
            use_hilbert_kernel=use_hilbert_kernel,
            use_perm=use_perm,
            SR=SR,
            X_embeds=point_embeds,
            data_type=data_type,
            load_embeds=False,
            fair_attr_df=point_fair,
            use_fair_perm=use_fair_perm,
            alpha=alpha,
            exec_per_point=True
        )
import numpy as np
import os
import cv2

from datetime import datetime
import random
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.utils import shuffle


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances_argmin_min

import pandas as pd

from tqdm import tqdm

from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import GPT2Tokenizer, GPT2Model

import torch

from transformers import pipeline
from transformers import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.set_verbosity_warning()

import itertools
import glob
import _pickle as pkl

from utils_dataShapley import create_model, calculate_error, scale_to_grid, merge_small_chunks
from utils import get_metric, transform_to_torch

class Data_Shapley_Dataset:

    def __init__(self, datasets_path, outpath, metric='Accuracy', 
                 problem='classification', 
                 model_type='MLP_Class', C=1, 
                 hidden_layers=[32, 32], 
                 num_of_classes=None):
        
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
        self.mc_ds = None
        self.vanilia_ds = None
        self.datasets_filenames = {}
        self.label_to_idx = {}
        self.use_img = False

    def read_data(self, class_label="class"):
        
        if os.path.exists(self.datasets_path):
            train_files = glob.glob(self.datasets_path + "/*.csv")
            datasets_train_df = {file_.split('/')[-1].split('.')[0].upper(): {'X': None, 'Y': None} for file_ in
                                 train_files}
            self.datasets_filenames = {file_.split('/')[-1].split('.')[0].upper(): file_ for file_ in train_files}
            for file_ in train_files:
                df = pd.read_csv(file_, index_col=False)
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
                for idx, row in df.iterrows():
                    X_lst.append(cv2.imread(row['file_path']))
                x_np = np.asarray(X_lst)
                datasets_train_dict[file_name]['X'] = x_np
                datasets_train_dict[file_name]['Y'] = class_labels
            return datasets_train_dict
        else:
            AssertionError(f'Filepath {self.datasets_path} did not exists!!')

    def prepare_chunks(self, df_dict, class_label="class"):
        out_dict = {data_name.split('/')[-1].split('.')[0].upper(): {'X': None, 'Y': None} for data_name, _ in df_dict.items()}
        
        for data_name, df in df_dict.items():
            data_name = data_name.split('/')[-1].split('.')[0].upper()
            class_labels = df.loc[:, df.columns.str.startswith(class_label)]
            if self.use_img:
                X_lst = []
                for idx, row in df.iterrows():
                    X_lst.append(cv2.imread(row['file_path']))
                x_np = np.asarray(X_lst)
                out_dict[data_name]['X'] = x_np
            else:
                row_tuples = df.drop(columns=list(class_labels.columns))
                out_dict[data_name]['X'] = row_tuples
            out_dict[data_name]['Y'] = class_labels

        return out_dict

    def save_data_chunks(self, file_dict, filename):
        out_path_dir = os.path.join(self.outpath, filename)
        self.datasets_path = out_path_dir
        if not os.path.exists(out_path_dir):
            print(f'Create new directory {out_path_dir}')
            os.mkdir(out_path_dir)
        dataset_filenames = []
        for file_name, df_ in file_dict.items():
            df_.to_csv(os.path.join(out_path_dir, file_name), index=False)
            dataset_filenames.append(os.path.join(out_path_dir, file_name))
            self.datasets_filenames[file_name.split('.')[0].upper()] = os.path.join(out_path_dir, file_name)

    def convert_data_with_pca(self, data, k_dim=3):
        pca = PCA(n_components=k_dim)
        data_tuples_embds = pca.fit_transform(data)
        return data_tuples_embds
    
    def convert_text_data_to_vectors(self, data, model_name='lamma'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2Model.from_pretrained(model_name, output_hidden_states=True).to(device)
            model.eval()

            # Add padding token if missing (GPT2 doesn’t have one)
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        elif model_name == 'lamma': 
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # quantized (GPTQ) version
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, output_hidden_states=True,).to(device)
            model.eval()
        elif model_name == 'bert':
            model_id = "bert-base-uncased"
        
            tokenizer = BertTokenizer.from_pretrained(model_id)
            model = BertModel.from_pretrained(model_id, output_hidden_states=True).to(device)

        if isinstance(data, dict):
            filenames = list(data.keys())
        elif isinstance(data, str):
            df = pd.read_csv(os.path.join(self.datasets_path, data))
            try:
                filenames = df['file_path'].tolist()
            except: 
                filenames = df['filename'].tolist()
        elif isinstance(data, pd.DataFrame):
            try:
                filenames = data['file_path'].tolist()
            except: 
                filenames = data['filename'].tolist()

        out_dict = {key: None for key in filenames}
        
        class_lst = []
        X_embed_lst = []
        for idx, filename in tqdm(enumerate(filenames)):
            with open(filename) as f: 
                txt_ = f.read()
            if len(txt_) == 0:
                txt_ = ' '

            inputs = tokenizer([txt_], padding=True, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():  
                outputs = model(**inputs)
                            
                if model_name == 'bert':
                    text_embeds = outputs.last_hidden_state[:, 0, :].flatten()    # CLS token representation
                else:
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    text_embeds = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
                    text_embeds = text_embeds.flatten()

                text_embeds = text_embeds.detach().cpu().numpy()
            
            out_dict[filename] = text_embeds
            class_lst.append(data.iloc[idx]['class'])
            X_embed_lst.append(text_embeds)
        columns = [f"X_{i+1}" for i in range(len(X_embed_lst[0]))]

        # Build dataframe
        out_df = pd.DataFrame(X_embed_lst, columns=columns)

        # Add class column
        out_df["class"] = class_lst
        return out_dict, out_df

    def convert_img_data_to_vectors(self, data):
        out_dict = {}
        pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-224", pool=True, device='cuda', use_fast=True, )
        if isinstance(data, dict):
            filenames = list(data.keys())
        elif isinstance(data, str):
            df = pd.read_csv(os.path.join(self.datasets_path, data))
            try:
                filenames = df['file_path'].tolist()
            except: 
                filenames = df['filename'].tolist()
        elif isinstance(data, pd.DataFrame):
            try:
                filenames = data['file_path'].tolist()
            except: 
                filenames = data['filename'].tolist()
            
            
        outputs = pipe(filenames)
        for embd, file_path in zip(outputs, filenames):
            out_dict[file_path] = np.asarray(embd[0], dtype=np.float64)
        
        return out_dict

    def split_by_cluster(self, chunk_size, data_tuples_embds):
        num_of_clusters = np.int_(np.ceil(data_tuples_embds.shape[0] / chunk_size))
        kmeans = KMeans(n_clusters=num_of_clusters, random_state=42)
        initial_labels = kmeans.fit_predict(data_tuples_embds)
        chunks = {}
        cluster_id = 0
        for label in np.unique(initial_labels):
            indices = np.where(initial_labels == label)[0]
            
            cluster_data = data_tuples_embds[indices]
            if len(indices) <= chunk_size:
               chunks[cluster_id] = indices
               cluster_id += 1 
            else:
                num_of_subclusters = np.int_(np.ceil(len(indices) / chunk_size))
                if num_of_clusters <= 1:
                    num_of_clusters = 2
                sub_kmeans = KMeans(n_clusters=num_of_subclusters, random_state=label)
                sub_labels = sub_kmeans.fit_predict(cluster_data)
                for sub_label in np.unique(sub_labels):
                    sub_indices = indices[sub_labels == sub_label]
                    chunks[cluster_id] = sub_indices
                    cluster_id += 1

        singletons = [cid for cid, idxs in chunks.items() if len(idxs) == 1]
        for singleton_cid in singletons:
            singleton_idx = chunks[singleton_cid][0]
            singleton_vec = data_tuples_embds[singleton_idx].reshape(1, -1)

            candidate_clusters = {cid: idxs for cid, idxs in chunks.items() if cid != singleton_cid}
            centroids = np.array([
                data_tuples_embds[idxs].mean(axis=0) for idxs in candidate_clusters.values()
            ])

            closest_ids, _ = pairwise_distances_argmin_min(singleton_vec, centroids)
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
        
        return chunks


    def split_by_tree(self, chunk_size, data_tuples_embds, use_query_r=False):
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
                    if use_query_r:
                        rad_ = 5
                        indxs = tree.query_radius(query_tuple, r=rad_)
                        if len(indxs[0]) > chunk_size:
                            indxs = indxs[:chunk_size]
                    else:
                        dist, indxs = tree.query(query_tuple, k=chunk_size)
                else:
                    chunks_idx[chunk_id] = idxs_
                    return chunks_idx
                
                chunks_idx[chunk_id] = [idxs_[i] for i in indxs[0]]
                data_tuples_embds_tmp = np.delete(data_tuples_embds_tmp, indxs[0], axis=0)
                idxs_ = [j for i, j in enumerate(idxs_) if i not in indxs[0]]

                chunk_id+=1

        if use_query_r:
            merge_small_chunks(data_tuples_embds, chunks_idx, chunk_size)
        return chunks_idx

    def update_img_fullpath(self, df):
        head_path, tail = os.path.split(self.datasets_path)
        for index, row in df.iterrows():
            df.at[index, 'file_path'] = os.path.normpath(os.path.join(head_path, row['file_path']))

        return df
    
    def calculate_num_of_classes(self, df, class_label="class"):
        class_labels = df.loc[:, df.columns.str.startswith(class_label)]
        class_unique = class_labels.drop_duplicates()[class_label].tolist()
        self.num_of_classes = len(class_unique)


    def labels_to_idx(self, df, class_label="class"):
        class_labels = df.loc[:, df.columns.str.startswith(class_label)]
        
        class_unique = class_labels.drop_duplicates()[class_label].tolist()
        self.label_to_idx = {label: i for i, label in enumerate(class_unique)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.num_of_classes = len(class_unique)

    def split_data(self, chunk_size, split_by='random', save_it=False, data_type='tabular', convert_labels=False, X_embeds=None):
        if os.path.exists(self.datasets_path):
            out_dict = {}
            if data_type.lower() == 'image' or data_type.lower() == 'img':
                self.use_img = True
                df_img_info = pd.read_csv(self.datasets_path, index_col=False)
                df_img_info = self.update_img_fullpath(df=df_img_info)
                file_name = self.datasets_path.split('/')[-1].split('.')[0].upper()
                self.calculate_num_of_classes(df=df_img_info)
                if convert_labels:
                    self.labels_to_idx(df=df_img_info)
                    df_img_info['class'] = df_img_info['class'].map(self.label_to_idx)
                X_embds_ret=None
                if split_by == 'random':
                    df_chunks = np.array_split(df_img_info, chunk_size)
                elif split_by.lower() == 'tree':
                    img_data = {row['file_path']: cv2.imread(row['file_path']) for index, row in df_img_info.iterrows()}
                    X_embds = self.convert_img_data_to_vectors(data=img_data)
                    X_embds = [embds for path_, embds in X_embds.items()]
                    X_embds = np.asarray(X_embds, dtype=np.float64)
                    chunk_idxs = self.split_by_tree(data_tuples_embds=X_embds, chunk_size=chunk_size)
                    df_chunks = [df_img_info.iloc[idxs] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                elif split_by.lower() == 'cluster':
                    img_data = {row['file_path']: cv2.imread(row['file_path']) for index, row in df_img_info.iterrows()}
                    X_embds = self.convert_img_data_to_vectors(data=img_data)
                    X_embds = [embds for path_, embds in X_embds.items()]
                    X_embds = np.asarray(X_embds, dtype=np.float64)
                    chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=X_embds)
                    df_chunks = [df_img_info.iloc[idxs] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]

                file_dict = {f'{file_name}_{i}.csv': df_ for i, df_ in enumerate(df_chunks)}
                if save_it:
                    self.save_data_chunks(file_dict=file_dict, filename=f'{file_name}_{chunk_size}')
                out_dict[file_name] = file_dict
                return out_dict, X_embds_ret
            elif data_type == 'tabular':
                self.use_img = False
                files = glob.glob(self.datasets_path + "/*.csv")
                out_dict = {}
                X_embds_ret=None
                for file in files:
                    df_file = pd.read_csv(file, index_col=False)
                    self.calculate_num_of_classes(df=df_file)
                    file_name = file.split('/')[-1].split('.')[0].upper()
                    X_embds_ret=None
                    df_file = df_file.fillna(-99)
                    if split_by == 'random':
                        df_chunks = np.array_split(df_file, chunk_size)
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
                        X_embds = self.convert_data_with_pca(data=df_file)
                        chunk_idxs = self.split_by_tree(data_tuples_embds=X_embds, chunk_size=chunk_size)
                        df_chunks = [df_file.iloc[idxs] for idxs in chunk_idxs.values()]
                        X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                    elif split_by.lower() == 'cluster':
                        df_file_np = np.asarray(df_file, dtype=np.float64)
                        chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=df_file_np)
                        df_chunks = [df_file.iloc[list(idxs)] for idxs in chunk_idxs.values()]
                        X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]

                    file_dict = {f'{file_name}_{i}.csv': df_ for i, df_ in enumerate(df_chunks)}
                    if save_it:
                        self.save_data_chunks(file_dict=file_dict, filename=f'{file_name}_{chunk_size}')
                    out_dict[file_name] = file_dict

                return out_dict, X_embds_ret
            elif data_type.lower() == 'text' or data_type.lower() == 'textual':
                self.use_img = False
                df_text_info = pd.read_csv(self.datasets_path, index_col=False)
                file_name = self.datasets_path.split('/')[-1].split('.')[0].upper()
                
                self.calculate_num_of_classes(df=df_text_info)

                if convert_labels:
                    self.labels_to_idx(df=df_text_info)
                    df_text_info['class'] = df_text_info['class'].map(self.label_to_idx)
                
                X_embds, df_text_info = self.convert_text_data_to_vectors(data=df_text_info)

                X_embds = [embds for _, embds in X_embds.items()]
                X_embds = np.asarray(X_embds, dtype=np.float64)
                if split_by == 'random':
                        df_chunks = np.array_split(df_text_info, chunk_size)
                        X_embds_ret = np.array_split(X_embds, chunk_size)
                elif split_by.lower() == 'tree':
                    chunk_idxs = self.split_by_tree(data_tuples_embds=X_embds, chunk_size=chunk_size)
                    df_chunks = [df_text_info.iloc[idxs] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                elif split_by.lower() == 'cluster':
                    chunk_idxs = self.split_by_cluster(chunk_size=chunk_size, data_tuples_embds=X_embds)
                    df_chunks = [df_text_info.iloc[list(idxs)] for idxs in chunk_idxs.values()]
                    X_embds_ret = [X_embds[idxs] for idxs in chunk_idxs.values()]
                
                file_dict = {f'{file_name}_{i}.csv': df_ for i, df_ in enumerate(df_chunks)}
                if save_it:
                    self.save_data_chunks(file_dict=file_dict, filename=f'{file_name}_{chunk_size}')
                out_dict[file_name] = file_dict

                return out_dict, X_embds_ret
            elif data_type.lower() == 'vectors' or data_type.lower() == 'vector embeddings':
                df_info = pd.read_csv(self.datasets_path, index_col=False)
                file_name = self.datasets_path.split('/')[-1].split('.')[0].upper()
                
                self.calculate_num_of_classes(df=df_info)
                X_embds_df = df_info.drop(columns=['class'])
                X_embds = X_embds_df.values.tolist()
                X_embds = np.asarray(X_embds, dtype=np.float64)

                if split_by == 'random':
                        df_chunks = np.array_split(df_info, chunk_size)
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

    def get_score(self, X, y, use_img=False, use_NN=False, lr=0.0001, max_epochs=1, batch_size=1, val_frac=0.1, data_type='tabular'):
        if data_type == 'text':
            use_norm = False
        else:
            use_norm = True

        self.model = create_model(model_type=self.model_type,problem=self.problem, max_epochs=max_epochs, 
                                  batch_size=batch_size, lr=lr, val_frac=val_frac, hidden_layer_size=self.hidden_layers, 
                                  use_img=use_img, use_NN=use_NN, num_classes=self.num_of_classes, norm=use_norm)
        if use_img:
            self.model.fit(X, y.values.ravel(), X_val=None, y_val=False, use_val=False)
            X_torch = transform_to_torch(X).to(self.model.device)
            y_pred = self.model.predict(X_torch)   
        elif use_NN:
            self.model.fit(X, y.values.ravel(), X_val=None, y_val=False, use_val=False)
            X_torch = transform_to_torch(X).to(self.model.device)
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
                
        return get_metric(metric=self.metric, y_pred=y_pred, y=y.values.ravel())

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

                        value = self.itter_Data_Shapley(X=X, y=y,
                                                        X_with_i=X_with_i, y_with_i=y_with_i,
                                                        n=n, subset=subset)

                        total_contribution += value
            shapley_values[i] = total_contribution

        Shap_values_dict = {key: shapley_values[i] for i, key in enumerate(data_dict.keys())}
        self.vanilia_ds = Shap_values_dict
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        return Shap_values_dict


    def itter_Data_Shapley(self, X, y, X_with_i, y_with_i, n, subset):
        perf_without_i = self.get_score(X=X, y=y, use_img=False, use_NN=False, lr=0.0001, max_epochs=1, batch_size=1, val_frac=0.1)

        perf_with_i = self.get_score(X=X_with_i, y=y_with_i, use_img=False, use_NN=False, lr=0.0001, max_epochs=1, batch_size=1, val_frac=0.1)

        weight = self.C / (n * (len(subset) + 1))
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

                value = self.itter_Data_Shapley(X=X, y=y,
                                                X_with_i=X_with_i, y_with_i=y_with_i,
                                                n=n, subset=subset)

                total_contribution += value
            shapley_values[i] = total_contribution

        Shap_values_dict = {key: shapley_values[i] for i, key in enumerate(data_dict.keys())}
        self.mc_ds = Shap_values_dict
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))
        return Shap_values_dict


    def init_score(self, metric, data_dict):
        if metric.lower() == 'accuracy':
            y = [data_['Y'] for data_ in list(data_dict.values())]
            y = pd.concat(y, axis=0)
            y = y.to_numpy().ravel()
            hist = np.bincount(y).astype(float)/len(y)
            return np.max(hist)

    def _tol_mean_score(self, data_dict):
        scores = []
        X = [data_['X'] for data_ in list(data_dict.values())]
        y = [data_['Y'] for data_ in list(data_dict.values())]

        model = create_model(self.model_type, self.hidden_layers, num_classes=self.num_of_classes)
        X_concat = pd.concat(X, axis=0)
        y_concat = pd.concat(y, axis=0)
        model.fit(X=X_concat, y=y_concat.values.ravel())

        for _ in range(100):
            bag_idxs = np.random.choice(len(X), len(y))
            x_bag_idxs = [data_['X'] for j, data_ in enumerate(list(data_dict.values())) if j in bag_idxs]
            y_bag_idxs = [data_['Y'] for j, data_ in enumerate(list(data_dict.values())) if j in bag_idxs]
            x_bag_idxs = pd.concat(x_bag_idxs, axis=0)
            y_bag_idxs = pd.concat(y_bag_idxs, axis=0)
            y_pred = model.predict(X=x_bag_idxs)

            scores.append(get_metric(metric=self.metric, y_pred=y_pred, y=y_bag_idxs))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
    
    def execute_per_dataset(self, split_data=False, chunk_size=200, k=50, split_by='random', use_hilbert_kernel=False, err=0.9,th=0.5, smart_selection=True, show_subsets=False, data_type='tabular', use_perm=True, ret_embeds=False, convert_label=False):
        # Check if we need to split data to chunks with chunk partitioning 
        if split_data:
            # Apply chunk partitioning and create chunks
            data, X_embed = self.split_data(chunk_size=chunk_size, split_by=split_by, save_it=True, data_type=data_type, convert_labels=convert_label)
            out_avg = {}
            out_shapley_values = {}
            selected_subsets = []
            # for each dataset perform data shapley with his chunks
            for file_name, df_dict in data.items():
                # prepare the chunks for the FLA-SHapley method
                df_dict_prepared = self.prepare_chunks(df_dict=df_dict, class_label="class")
                if show_subsets:
                    Shap_values_dict, subsets_ = self.FLA_SHapley(data_dict=df_dict_prepared, df_dict=df_dict, err=0.9, total_subset=k, use_gradient=True, chunk_size=None, th=th, smart_selection=smart_selection, show_subsets=show_subsets, use_hilbert_kernel=use_hilbert_kernel, X_embeds=X_embed, use_perm=use_perm)
                    selected_subsets += subsets_
                else:
                    Shap_values_dict = self.FLA_SHapley(data_dict=df_dict_prepared, df_dict=df_dict, err=0.9, total_subset=k, use_gradient=True, chunk_size=None, th=th, smart_selection=smart_selection, show_subsets=show_subsets, use_hilbert_kernel=use_hilbert_kernel, X_embeds=X_embed, use_perm=use_perm)
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
            # Did the same for a single datasets which has already perform chunk partitioning
            print('Start C-DaSh Run on pre-divided dataset in chunk')
            if data_type.lower() == 'image' or data_type.lower() == 'img':
                data_dict = self.read_img_data()
            else:
                data_dict = self.read_data()
            selected_subsets = []
            if show_subsets:
                Shap_values_dict, subsets_ = self.FLA_SHapley(data_dict=data_dict, err=0.9, total_subset=k, use_gradient=True, chunk_size=None, 
                                                                       th=th, smart_selection=smart_selection, show_subsets=show_subsets, 
                                                                       use_hilbert_kernel=use_hilbert_kernel, X_embeds=None, use_perm=use_perm)
                selected_subsets += subsets_
            else:
                Shap_values_dict = self.FLA_SHapley(data_dict=data_dict, err=0.9, total_subset=k, use_gradient=True, chunk_size=None, th=th, 
                                                             smart_selection=smart_selection, show_subsets=show_subsets, use_hilbert_kernel=use_hilbert_kernel, 
                                                             X_embeds=None, use_perm=use_perm)
            if show_subsets:
                print(f'Selected subsets : ')
                print(selected_subsets)
            dataset_avg = np.mean(list(Shap_values_dict.values()))
            self.values_ds_FLA_SH_run = Shap_values_dict
            if ret_embeds:
                return out_shapley_values, out_avg, X_embed
            return Shap_values_dict, dataset_avg

    def subset_selection(self, data_dict, data_X, data_y, subsets, lr, total_subset, th,  metrics, data_type, show_subsets=False, val_frac=0.1, use_gradient=True, use_norm=True):
        data_flags = {key: True for key in data_dict.keys()}
        data_count = {key: 0 for key in data_dict.keys()}

        data_dict_keys_lst = list(data_dict.keys())

        allowed_perc = np.ceil(total_subset * 0.25)
        priority_counter = 5
        # For each subset check if we are sattisfied it or we continue with a new one
        for idx in range(len(subsets)):
            flag = False
            # check if in a chunk a data point is more than the allowed percentage to be in a subset
            for chunk_id in subsets[idx]:
                if data_count[data_dict_keys_lst[chunk_id]] >= allowed_perc:
                    flag = True

            attempts = 0
            if self.problem == 'regression':
                eq_ = lambda a, b: a > b 
            else:
                eq_ = lambda a, b: a < b 

            while (eq_(metrics[idx], th) or flag):
                # if repeat more than 5 times for a subset with index idx exit (to not get in infinitive loop)
                if attempts >= 5:
                    break

                attempts += 1
                # Select a new subset and we guide to select only in the first palces which is the most probably high quality chunks
                if idx == len(data_dict_keys_lst):
                    new_subset = list(set(random.sample(range(idx-2), random.randint(0, idx-2))))
                    while len(new_subset) == 0:
                        new_subset = list(set(random.sample(range(idx-2), random.randint(0, idx-2))))
                else:
                    new_subset = list(set(random.sample(range(idx+2), random.randint(0, idx+2))))
                    while len(new_subset) == 0:
                        new_subset = list(set(random.sample(range(idx+2), random.randint(0, idx+2))))
                
                for j, chunk_id in enumerate(new_subset):
                    data_keys = data_dict_keys_lst[chunk_id]
                    
                    # check the remain valid canditates chunks where sattisfied the allowed percentage to be in a chunk
                    valid_candidates = [i for i, k in enumerate(data_dict_keys_lst) if data_count[k] < allowed_perc]
                    if len(valid_candidates) <= 1:
                        break
                    if data_count[data_keys] >= allowed_perc:
                        if len(valid_candidates) == 0:
                            new_subset[j] = random.randint(0, idx+priority_counter)
                        else:
                            new_subset[j] = random.choice(valid_candidates)

                flag = False

                new_subset = set(new_subset)
                subsets[idx] = new_subset
                X = [data_['X'] for j, data_ in enumerate(list(data_dict.values())) if j in new_subset]
                y = [data_['Y'] for j, data_ in enumerate(list(data_dict.values())) if j in new_subset]
                
                if use_gradient:
                    if all(isinstance(item, pd.DataFrame) for item in X):
                        X = pd.concat(X, axis=0)
                        y = pd.concat(y, axis=0)
                    elif all(isinstance(item, np.ndarray) for item in X):
                        X = np.concatenate(X, axis=0)
                        y = np.concatenate(y, axis=0)
                    
                    data_X[idx] = X
                    data_y[idx] = y

                    if len(X) * val_frac < 2:
                        val_fracs = np.arange(0.01, val_frac, 0.01, dtype=np.float64)
                        for frac in val_fracs:
                            if len(X) * val_frac > 2:
                                val_frac = frac
                    
                    model = create_model(model_type=self.model_type, 
                                 problem=self.problem,
                                 max_epochs=1,
                                 batch_size=1, 
                                 lr=lr, 
                                 val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=self.use_img, 
                                 use_NN=True,
                                 num_classes=self.num_of_classes, 
                                 norm=use_norm)
                    model.fit(X=data_X, y=data_y, X_val=None, y_val=None, data_dict=None, batch_size=1, epochs=1, metric=self.metric)
                    new_metric = model.history['metrics'][idx] 
                else:
                    new_metric = self.get_score(X=X, y=y, use_img=self.use_img, use_NN=False, lr=lr, max_epochs=1, batch_size=1, val_frac=0.1, data_type=data_type)
                
                # if the new chunk has better inidvidual contribution we update it
                if eq_(metrics[idx], new_metric):
                    subsets[idx] = new_subset
                    metrics[idx] = new_metric

            for chunk_id in subsets[idx]:
                data_count[data_dict_keys_lst[chunk_id]] += 1
                if data_count[data_dict_keys_lst[chunk_id]] > allowed_perc:
                    data_flags[data_dict_keys_lst[chunk_id]] = False
                    priority_counter += 1
                    if priority_counter >= len(data_count):
                        priority_counter = len(data_count) - 1
            
        return subsets, metrics
    
    def FLA_SHapley(self, data_dict=None, df_dict=None, err=0.9, total_subset=50, use_gradient=False, lr=0.001, chunk_size=None, th=0.6, smart_selection=True, val_frac=0.1, show_subsets=False, use_hilbert_kernel=False, use_perm=True, SR=0.1, X_embeds=None, data_type='tabular', load_embeds=False):
        sr_shapley = True
        
        if data_type == 'text':
            use_norm = False
        else:
            use_norm = True

        if data_dict is None:
            data, X_embeds = self.split_data(chunk_size=chunk_size, split_by='', save_it=True, data_type=data_type, convert_labels=False)
            data_dict = self.prepare_chunks(df_dict=data, class_label="class")
        elif not isinstance(data_dict, dict):
            data_dict = self.read_data()

        n = len(data_dict)
        if n <=1:
            Shap_values_dict = {key: 0 for i, key in enumerate(data_dict.keys())}
            self.values_ds_FLA_SH_run = Shap_values_dict
            return Shap_values_dict
        if n < total_subset:
            print(f'Change subset size to the length of the chunks to {n/2}, because cannot be the chunk size k > n')
            total_subset = int(n/2)
        
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
                    for filename, chunk in df_dict.items():
                        embeds = self.convert_img_data_to_vectors(data=filename)
                        data_dim.append(np.mean(list(embeds.values()), axis=0))
                    data_dim = np.asarray(data_dim)
                    dim = data_dim[0].shape[0]
                else:
                    dim = X_embeds[0].shape[1]
                    
                    data_dim = []
                    for arr in X_embeds:
                        data_dim.append(np.mean(arr, axis=0))
            elif data_type.lower() == 'text' or data_type.lower() == 'textual':
                data_dim = []
                if X_embeds == None:
                    X_embeds_lst = []
                    if load_embeds:
                        for filename, chunk in df_dict.items():
                            X_embds = chunk['X']
                            data_dim.append(np.mean(list(X_embds), axis=0))
                            X_embeds_lst.append(X_embeds)
                        data_dim = np.asarray(data_dim)
                        dim = data_dim[0].shape[0]
                        X_embeds = X_embeds_lst 
                    else:
                        for filename, chunk in df_dict.items():
                            X_embds, df_text_info = self.convert_text_data_to_vectors(data=filename)
                            data_dim.append(np.mean(list(X_embds.values()), axis=0))
                            X_embeds_lst.append(X_embeds)
                        data_dim = np.asarray(data_dim)
                        dim = data_dim[0].shape[0]
                        X_embeds = X_embeds_lst
                else:
                    dim = X_embeds[0].shape[1]
                    
                    data_dim = []
                    for arr in X_embeds:
                        data_dim.append(np.mean(arr, axis=0))
                    
            elif data_type.lower() == 'tabular':
                pca = PCA(n_components=dim) 
                data_dim = [pd.concat([data_['X'], data_['Y']], axis=1).to_numpy() for name, data_ in data_dict.items()]
                
                data_dim = [np.mean(pca.fit_transform(arr), axis=0) for arr in data_dim]
            data_names = [name for name, data_ in data_dict.items()]
            data_dim = np.vstack(data_dim)

            # Scale points to grid and compute the Hilbert curve to permute the data by their most possible importance
            scaled_positions = scale_to_grid(data_dim, hilbert_bits)

            hilbert_curve = HilbertCurve(hilbert_bits, dim)
            
            hilbert_indices = np.array([hilbert_curve.distance_from_point(coord.tolist()) for coord in scaled_positions])

            objects = list(zip(data_names, data_dim, priorities, hilbert_indices))

            sorted_chunked = sorted(objects, key=lambda x: x[3], reverse=False)
            data_dict_tmp = {name: data_dict[name] for name, dim, priority, hilbert_index in sorted_chunked}
            data_dict = data_dict_tmp
            print('Finish Hilbert Permutation')
        elif use_perm:
            print('Start SR permutation')
            scores_tmp = {key: 0 for key, values in data_dict.items()}
            # For each data point get his individual condribution on the performance metric score
            for key, values in data_dict.items():
                random_indices = random.sample(range(values['X'].shape[0]), np.ceil(values['X'].shape[0]*SR).astype(np.int_))
                
                if isinstance(values['X'], pd.DataFrame):
                    X_ = values['X'].iloc[random_indices]
                else:
                    X_ = values['X'][random_indices]
                
                if isinstance( values['Y'], pd.DataFrame):
                    y_ =  values['Y'].iloc[random_indices]
                else:
                    y_ =  values['Y'][random_indices]

                score_ = self.get_score(X=X_, y=y_, use_img=self.use_img, use_NN=True, lr=lr, max_epochs=10, batch_size=1, val_frac=val_frac, data_type=data_type) 
                scores_tmp[key] = score_
            # Sort the data by their individual contribution
            scores_tmp = dict(sorted(scores_tmp.items(), key=lambda item: item[1], reverse=True))
            data_dict_tmp = {key: data_dict[key] for key, value in scores_tmp.items()}
            data_dict = data_dict_tmp
            print('Finish Permutation')
        
        memory_ds = np.zeros((n, total_subset))
        metrics = []
        itter_count = 0
        while sr_shapley:
            # Randomly select k subset
            subsets = [set(random.sample(range(i), random.randint(0, i))) for i in range(2, total_subset+2)]
            print(f'Itteration {itter_count+1}')
            
            if use_gradient:
                data_X = []
                data_y = []

            # Find the contribution of each subset in the performance metric and the model A
            for idx, subset in enumerate(tqdm(subsets)):
                while len(subset) == 0:
                    subset = set(random.sample(range(idx+2), random.randint(0, idx+2)))
                    subsets[idx] = subset
                    
                X = [data_['X'] for j, data_ in enumerate(list(data_dict.values())) if j in subset]
                y = [data_['Y'] for j, data_ in enumerate(list(data_dict.values())) if j in subset]
                if all(isinstance(item, pd.DataFrame) for item in X):
                    X = pd.concat(X, axis=0)
                    y = pd.concat(y, axis=0)
                elif all(isinstance(item, np.ndarray) for item in X):
                    X = np.concatenate(X, axis=0)
                    y = np.concatenate(y, axis=0)

                if use_gradient:
                    data_X.append(X)
                    data_y.append(y)
                    metrics.append(0)
                else:
                    metrics.append(self.get_score(X=X_, y=y_, use_img=self.use_img, use_NN=False, lr=lr, max_epochs=1, batch_size=1, val_frac=0.1, data_type=data_type))
            if use_gradient:

                model = create_model(model_type=self.model_type, 
                                         problem=self.problem,
                                         max_epochs=1,
                                         batch_size=1, 
                                         lr=lr, 
                                         val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=self.use_img, 
                                         use_NN=True,
                                         num_classes=self.num_of_classes,
                                         norm=use_norm)
                model.fit(X=data_X, y=data_y, X_val=None, y_val=None, data_dict=None, batch_size=1, epochs=1, metric=self.metric)

                for i, index in enumerate(model.history['idxs']):
                    metrics[index] = model.history['metrics'][i]                
            
            # Check the performance of each subset and apply optimised subset selection by select the subset that satisfied a threshold th
            if smart_selection:
                subsets, metrics = self.subset_selection(data_dict=data_dict, data_X=data_X, data_y=data_y, subsets=subsets, lr=lr, total_subset=total_subset, th=th,  
                                                         metrics=metrics, show_subsets=show_subsets, val_frac=val_frac, use_norm=use_norm, data_type=data_type)

            if show_subsets:
                subset_names = []
                for subset in subsets:
                    for k, data_name in enumerate(list(data_dict.keys())):
                        if k in subset:
                            subset_names.append(data_name)
            
            marginal_contrib = np.zeros((n, total_subset))

            model = create_model(model_type=self.model_type, 
                                         problem=self.problem,
                                         max_epochs=1,
                                         batch_size=1, 
                                         lr=lr, 
                                         val_frac=0.1, hidden_layer_size=self.hidden_layers, use_img=self.use_img, 
                                         use_NN=True,
                                         num_classes=self.num_of_classes,
                                         norm=use_norm)
            for i in tqdm(range(n)):
                for j, subset in enumerate(subsets):
                    if i in subset or len(subset) == 0:
                        # If subset length equal with zero or i in subset we continue to the next one
                        continue
                    # Find the score for the subset with and without chunk with index i
                    subset_with_i = subset | {i}
                    X = [data_['X'] for k, data_ in enumerate(list(data_dict.values())) if k in subset_with_i]
                    y = [data_['Y'] for k, data_ in enumerate(list(data_dict.values())) if k in subset_with_i]
                    if all(isinstance(item, pd.DataFrame) for item in X):
                        X = pd.concat(X, axis=0)
                        y = pd.concat(y, axis=0)
                    elif all(isinstance(item, np.ndarray) for item in X):
                        X = np.concatenate(X, axis=0)
                        y = np.concatenate(y, axis=0)
                    
                    if use_gradient:
                        model.fit(X=X, y=y, X_val=None, y_val=None, data_dict=None, batch_size=1, epochs=1, metric=self.metric)
                        perf_with_i = model.history['metrics'][-1]
                    else:
                        perf_with_i = self.get_score(X=X_, y=y_, use_img=self.use_img, use_NN=False, lr=lr, max_epochs=100, batch_size=1, val_frac=0.1, data_type=data_type)

                    # For each data point find the Shapley value among all the available subsets
                    weight = self.C / (n * (len(subset) + 1))
                    value = weight * (perf_with_i - metrics[j])
                    marginal_contrib[i, j] = value
            
            if itter_count == 0:
                memory_ds = marginal_contrib
            else:
                memory_ds = np.concatenate([memory_ds, marginal_contrib]) 
            
            print(f'Trancation error at iteration {itter_count} : {calculate_error(memory_ds)}')
            if calculate_error(memory_ds) < err:
                break
            itter_count += 1

        print(f'Final Trancation error : {calculate_error(memory_ds)}')
        Shap_values_dict = {key: np.mean(memory_ds[i]) for i, key in enumerate(data_dict.keys())}
        self.values_ds_FLA_SH_run = Shap_values_dict
        
        if show_subsets:
            return Shap_values_dict, subset_names
        else:
            return Shap_values_dict

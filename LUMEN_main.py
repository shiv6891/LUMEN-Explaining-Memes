#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torch import optim, nn
from torchvision import models, transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from PIL import Image

import cv2

import time

from transformers import BertModel
from transformers import AdamW

# from tqdm.auto import tqdm
from tqdm import tqdm,trange 
# from tqdm import trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import  mean_absolute_error
from transformers.optimization import Adafactor
# from tabulate import tabulate
import os, sys
sys.path.append('/path-to/early-stopping-pytorch')
from pytorchtools import EarlyStopping
import glob

import math
# % matplotlib inline
import os
torch.cuda.set_device(1)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
CUDA_LAUNCH_BLOCKING=1
from transformers.utils import logging
logging.set_verbosity(40)


# In[4]:


from transformers import ViTFeatureExtractor, ViTModel
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
deBERTatokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# --T5
T5tokenizer = T5Tokenizer.from_pretrained("t5-large")

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)



# In[16]:


# Default dataset files
# import pandas as pd
data_dir = '/path-to/images'
train_path = '/path-to/hvvexp_train.csv'
dev_path = '/path-to/hvvexp_val.csv'
test_path = '/path-to/hvvexp_test.csv'




# T5 decoder
class HarmemeMemesDatasetAug(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        mode=None,
#         image_transform,
#         text_transform,
        balance=False,
        dev_limit=None,
        random_state=0,
    ):
        self.mode = mode

        # self.samples_frame = pd.read_json(
        #     data_path, lines=True
        self.samples_frame = pd.read_csv(
                data_path, index_col=0
            )
     

        self.samples_frame = self.samples_frame.reset_index(
            drop=True
        )
        
        # print(self.samples_frame.head())
        self.samples_frame.image = self.samples_frame.apply(
            lambda row: (img_dir + '/' + row.image), axis=1
        )

    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_id = self.samples_frame.loc[idx, "id"]
        img_name = self.samples_frame.loc[idx, "image"]  
        # print(f'img_name: {img_name}')

    
#         ***Get VIT input data***
        file_name = self.samples_frame.loc[idx, "image"]
        vit_image_data = Image.open(file_name)
        if vit_image_data.mode != 'RGB':
            vit_image_data = vit_image_data.convert('RGB')  
        vit_image_data = feature_extractor(vit_image_data, return_tensors="pt")
        
        
        ocr = self.samples_frame.loc[idx, "OCR"]
        ent = self.samples_frame.loc[idx, "entity"]
        role = self.samples_frame.loc[idx, "role"]
        caption = self.samples_frame.loc[idx, "caption"]
        
        bert_inputs_ocr = ocr
        bert_inputs_ent = ent
        
        
        bert_inputs = [bert_inputs_ocr, bert_inputs_ent]
        
               
        
        # ---------------------------------------------
        # T5 douple scenario: prompt input + caption
        T5_source1 = "Generate explanation for "+ent+" as "+role+": "+ocr.replace('\n', ' ').replace(' .', '.')
        T5_source2 = caption
       
        
        if self.mode != 'test':
            T5_target = self.samples_frame.loc[idx, "explanation"]
        else:
            T5_target = self.samples_frame.loc[idx, "siddhant's explanations"]
        if self.samples_frame.loc[idx, "role"]=="hero":
            lab=0
        elif self.samples_frame.loc[idx, "role"]=="victim":
            lab=1     
        else:
            lab=2
        label = torch.tensor(lab).to(device)  

        sample = {
            # "id": img_id, 
            "img_name": img_name,                
            "bert_inputs": bert_inputs,
            # "inputs_ocr": bert_inputs_ocr,
            # "inputs_entity": bert_inputs_ent,
            "vit_image_data": vit_image_data,
            # "det_img_bgr": img_bgr,
            "label": label,
            "T5_source1": T5_source1,
            "T5_source2": T5_source2,
            "T5_target": T5_target
        }
        return sample


# In[28]:


BS = 4 #at least 10 can be tried (12327MiB being used)

hm_dataset_train = HarmemeMemesDatasetAug(train_path, data_dir)
dataloader_train = DataLoader(hm_dataset_train, batch_size=BS,
                        shuffle=True, num_workers=0)
hm_dataset_val = HarmemeMemesDatasetAug(dev_path, data_dir)
dataloader_val = DataLoader(hm_dataset_val, batch_size=BS,
                        shuffle=True, num_workers=0)
hm_dataset_test = HarmemeMemesDatasetAug(test_path, data_dir, mode='test')
dataloader_test = DataLoader(hm_dataset_test, batch_size=BS,
                        shuffle=False, num_workers=0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
data_time = AverageMeter('Data', ':6.3f')


from pathlib import Path
class MM(nn.Module):
    def __init__(self, n_out):
        super(MM, self).__init__()               
        self.model_ViT = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model_deBERTa = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge", num_labels=n_out, problem_type="multi_label_classification", output_hidden_states=True)
        self.model_T5 = T5ForConditionalGeneration.from_pretrained("t5-large")
        
        self.trans1 = nn.Linear(768,512)
        self.trans2 = nn.Linear(1536,512)
        self.trans3 = nn.Linear(1024,512)        
        
        self.lin1 = nn.Linear(1536,512)
        self.out = nn.Linear(512,n_out)
        
    # vit_inputs, deBERTainputs, deBERTalabels, T5input_ids, T5attention_mask, T5labels
    def forward(self, vit_inputs, dinputs, dlabels, T5input_ids, T5attention_mask, T5labels):
        # print("inside the forward loop")
        vit_output = self.model_ViT(vit_inputs)
        vit_pooled_out = vit_output.pooler_output
        
        deBERTa_output = self.model_deBERTa(**dinputs, labels=dlabels)
        deBERTa_lasthid = deBERTa_output.hidden_states[-1]
        deBERTa_pooled_out = torch.mean(deBERTa_lasthid, 1)
        
        T5_output = self.model_T5(input_ids=T5input_ids, attention_mask=T5attention_mask, labels=T5labels, output_hidden_states=True, return_dict=True)
        T5_lasthid = T5_output.decoder_hidden_states[-1]
        T5_pooled_out = torch.mean(T5_lasthid, 1)
        
        # Transform
        vit_mm = F.relu(self.trans1(vit_pooled_out))
        dberta_mm = F.relu(self.trans2(deBERTa_pooled_out))
        t5_mm = F.relu(self.trans3(T5_pooled_out))
        
        # Additional components        
        
        # vbert_pooled_out = torch.mean(vbert_encoder_lasthid, 1)
        fused_out = torch.cat([vit_mm, dberta_mm, t5_mm], axis=1)
        out1 = F.relu(self.lin1(fused_out))
        out = self.out(out1)
        return T5_output, out, deBERTa_output.loss
# ----------------------------------------------
# Can comment out the following to run separately

try:
    del model
except:
    pass
# output_size = 1 #Binary case
output_size = 3
model = MM(output_size)
model.to(device)


code_prof = False

exp_name = "name-of-experiment"
exp_path = "/path-to-saved-model/"+exp_name

criterion = nn.CrossEntropyLoss()

# replace AdamW with Adafactor
optimizer = Adafactor(
    model.parameters(),
    lr=1e-4,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)


max_source_length = 512
max_target_length = 512

# For cross entropy loss
def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5
    
    train_acc_list=[]
    val_acc_list=[]
    train_loss_list=[]
    val_loss_list=[]
    train_T5encdec_loss_list=[]
    val_T5encdec_loss_list=[]
    train_main_loss_list=[]
    val_main_loss_list=[]
    train_deBERTa_loss_list=[]
    val_deBERTa_loss_list=[]
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)
    

    model.train()
    for i in range(epochs):
        # print(f"******************************EPOCH - {i}****************************************")
#         total_acc_train = 0
        total_loss_train = 0
        total_T5encdec_loss_train = 0
        total_main_loss_train = 0
        total_deBERTa_loss_train = 0
        total_train = 0
        correct_train = 0
        # for data in dataloader_train:
        for data in tqdm(dataloader_train, total = len(dataloader_train), desc = f"Mini-batch progress (Train) | Epoch: {i+1}"):
            # print(f'------------------Mini Batch - {mbcnt+1}------------------')
            # mbcnt+=1
            
            
            
            pixel_values_start = time.time()
            vit_inputs = data['vit_image_data'].pixel_values.squeeze().to(device)
            data_time.update(time.time() - pixel_values_start)
            if code_prof:
                print(f"vision_inputs processing time: {data_time.val}")
                
                
            # deBERTa inputs
            deBERTainputs = deBERTatokenizer(data['bert_inputs'][0], data['bert_inputs'][1], padding=True, return_tensors="pt").to(device)
            deBERTalabels = torch.nn.functional.one_hot(data['label'],num_classes=3).to(torch.float).to(device)
            

            data_time.reset()
            decoder_labels_start = time.time()
            
            T5encoding = T5tokenizer(
                data['T5_source1'],
                data['T5_source2'],
                padding="longest",
                max_length=max_source_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            T5input_ids, T5attention_mask = T5encoding.input_ids, T5encoding.attention_mask
            
            # encode the targets
            T5target_encoding = T5tokenizer(
                data['T5_target'], padding="longest", max_length=max_target_length, truncation=True
            )
            
            T5labels = T5target_encoding.input_ids
            # replace padding token id's of the labels by -100 so it's ignored by the loss
            T5labels = torch.tensor(T5labels).to(device)
            T5labels[T5labels == T5tokenizer.pad_token_id] = -100
            if code_prof:
                print(f"T5 input processing time: {data_time.val}")
            
            label = data['label'].to(device)
            
            model.zero_grad()
            data_time.reset()
            model_start = time.time()
            
            T5encdec_out, main_out, deBERTa_loss = model(vit_inputs, deBERTainputs, deBERTalabels, T5input_ids, T5attention_mask, T5labels)
            data_time.update(time.time() - model_start)
            if code_prof:
                print(f"model processing time: {data_time.val}")
            # print(vencdec_out.decoder_hidden_states[-1].shape)
            T5encdec_loss = T5encdec_out.loss
            main_loss = criterion(main_out.squeeze(), label)
            # print(main_loss)
            loss = 0.5*T5encdec_loss+0.3*main_loss+0.2*deBERTa_loss
            # print(f"vencdec_loss: {vencdec_loss.item()} | main_loss: {main_loss.item() | Total loss: {loss.item()}}")
            loss.backward()
            optimizer.step()            
            # print(main_out.data)
            with torch.no_grad():
                # print(torch.max(main_out.data, 1))
                _, predicted_train = torch.max(main_out.data, 1)
                total_train += label.size(0)
                correct_train += (predicted_train == label).sum().item()
                total_T5encdec_loss_train += T5encdec_loss.item()
                total_main_loss_train += main_loss.item()
                total_deBERTa_loss_train += deBERTa_loss.item()
                total_loss_train += loss.item()
            # break

        # break
        train_acc = 100 * correct_train / total_train
        train_loss = total_loss_train/total_train
        train_T5encdec_loss = total_T5encdec_loss_train/total_train
        train_main_loss = total_main_loss_train/total_train
        train_deBERTa_loss = total_deBERTa_loss_train/total_train
        model.eval()
#         total_acc_val = 0
        total_loss_val = 0
        total_T5encdec_loss_val = 0
        total_main_loss_val = 0
        total_deBERTa_loss_val = 0
        total_val = 0
        correct_val = 0

        with torch.no_grad():
            for data in tqdm(dataloader_val, total = len(dataloader_val), desc = "Mini-batch progress (Val)"):               
                pixel_values_start = time.time()
                vit_inputs = data['vit_image_data'].pixel_values.squeeze().to(device)
                data_time.update(time.time() - pixel_values_start)
                if code_prof:
                    print(f"vision_inputs processing time: {data_time.val}")


                # deBERTa inputs
                deBERTainputs = deBERTatokenizer(data['bert_inputs'][0], data['bert_inputs'][1], padding=True, return_tensors="pt").to(device)
                deBERTalabels = torch.nn.functional.one_hot(data['label'],num_classes=3).to(torch.float).to(device)


                data_time.reset()
                decoder_labels_start = time.time()
                T5encoding = T5tokenizer(
                    data['T5_source1'],
                    data['T5_source2'],
                    padding="longest",
                    max_length=max_source_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                T5input_ids, T5attention_mask = T5encoding.input_ids, T5encoding.attention_mask

                # encode the targets
                T5target_encoding = T5tokenizer(
                    data['T5_target'], padding="longest", max_length=max_target_length, truncation=True
                )

                T5labels = T5target_encoding.input_ids
                # replace padding token id's of the labels by -100 so it's ignored by the loss
                T5labels = torch.tensor(T5labels).to(device)
                T5labels[T5labels == T5tokenizer.pad_token_id] = -100
                if code_prof:
                    print(f"T5 input processing time: {data_time.val}")

                label_val = data['label'].to(device)
                model.zero_grad()
                data_time.reset()
                model_start = time.time()
                T5encdec_out_val, main_out_val, deBERTa_loss_val = model(vit_inputs, deBERTainputs, deBERTalabels, T5input_ids, T5attention_mask, T5labels)
                data_time.update(time.time() - model_start)
                if code_prof:
                    print(f"model processing time: {data_time.val}")
                # print(main_out_val.squeeze())
                T5encdec_loss_val = T5encdec_out_val.loss
                main_loss_val = criterion(main_out_val.squeeze(), label_val)
                # print(main_loss_val)
                loss_val = 0.5*T5encdec_loss_val+0.3*main_loss_val+0.2*deBERTa_loss_val
                
                
                _, predicted_val = torch.max(main_out_val.data, 1)
                total_val += label_val.size(0)
                correct_val += (predicted_val == label_val).sum().item()                
                total_T5encdec_loss_val += T5encdec_loss_val.item()
                total_main_loss_val += main_loss_val.item()
                total_deBERTa_loss_val += deBERTa_loss_val.item()
                total_loss_val += loss_val.item()
                                
                
        print("Saving model...") 
        torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))

        val_acc = 100 * correct_val / total_val
        val_loss = total_loss_val/total_val
        val_T5encdec_loss = total_T5encdec_loss_val/total_val
        val_main_loss = total_main_loss_val/total_val
        val_deBERTa_loss = total_deBERTa_loss_val/total_val
        
        
        
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_T5encdec_loss_list.append(train_T5encdec_loss)
        val_T5encdec_loss_list.append(val_T5encdec_loss)
        train_main_loss_list.append(train_main_loss)
        val_main_loss_list.append(val_main_loss)
        train_deBERTa_loss_list.append(train_deBERTa_loss)
        val_deBERTa_loss_list.append(val_deBERTa_loss)
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        print(f'Epoch {i+1}: train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | train_T5encdec_loss: {train_T5encdec_loss:.4f} | val_T5encdec_loss: {val_T5encdec_loss:.4f} | train_main_loss: {train_main_loss:.4f} | val_main_loss: {val_main_loss:.4f} | train_deBERTa_loss: {train_deBERTa_loss:.4f} | val_deBERTa_loss: {val_deBERTa_loss:.4f}')
        with open(os.path.join(exp_path, exp_name+'_base_exp_results.txt'), 'a+') as of:
            of.write(f'Epoch {i+1}: train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | train_T5encdec_loss: {train_T5encdec_loss:.4f} | val_T5encdec_loss: {val_T5encdec_loss:.4f} | train_main_loss: {train_main_loss:.4f} | val_main_loss: {val_main_loss:.4f} | train_deBERTa_loss: {train_deBERTa_loss:.4f} | val_deBERTa_loss: {val_deBERTa_loss:.4f}\n')
        
        model.train()
        torch.cuda.empty_cache()
        
    return  model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, train_T5encdec_loss_list, val_T5encdec_loss_list, train_main_loss_list, val_main_loss_list, train_deBERTa_loss, val_deBERTa_loss, i
        

    
train = False

if train:
    n_epochs = 15
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 15
    model, train_acc_list, val_acc_list, train_loss_list, val_loss_list, train_T5encdec_loss_list, val_T5encdec_loss_list, train_main_loss_list, val_main_loss_list, train_deBERTa_loss, val_deBERTa_loss, i = train_model(model, patience, n_epochs)


# For T5 based model
def test_model(model):
    model.eval()
    code_prof = False
#         total_acc_val = 0
    total_loss_test = 0
    total_vencdec_loss_test = 0
    total_main_loss_test = 0
    total_deBERTa_loss_test = 0
    total_test = 0
    correct_test = 0
    generated_result = []
    predicted_label_list = []
    true_label_list = []
    img_list = []

    with torch.no_grad():
        # for data in dataloader_test:  
        for data in tqdm(dataloader_test, total = len(dataloader_test), desc = "Mini-batch progress (Test)"):  
            cur_imgs = [x.split('/')[-1] for x in data['img_name']]
            img_list+=cur_imgs
            # print(cur_imgs)
            pixel_values_start = time.time()
            if len(data['vit_image_data'].pixel_values.squeeze().size())<BS:
                # print(data['vit_image_data'].pixel_values.squeeze(0).shape)
                vit_inputs = data['vit_image_data'].pixel_values.squeeze(0).to(device)
            else:
                vit_inputs = data['vit_image_data'].pixel_values.squeeze().to(device)
            # print(vit_inputs.shape)
            data_time.update(time.time() - pixel_values_start)
            if code_prof:
                print(f"pixel_values processing time: {data_time.val}")

            # deBERTa inputs
            deBERTainputs = deBERTatokenizer(data['bert_inputs'][0], data['bert_inputs'][1], padding=True, return_tensors="pt").to(device)
            deBERTalabels = torch.nn.functional.one_hot(data['label'],num_classes=3).to(torch.float).to(device)

            data_time.reset()
            decoder_labels_start = time.time()
            T5encoding = T5tokenizer(
                data['T5_source1'],
                data['T5_source2'],
                padding="longest",
                max_length=max_source_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            T5input_ids, T5attention_mask = T5encoding.input_ids, T5encoding.attention_mask

            # encode the targets
            T5target_encoding = T5tokenizer(
                data['T5_target'], padding="longest", max_length=max_target_length, truncation=True
            )

            T5labels = T5target_encoding.input_ids
            # replace padding token id's of the labels by -100 so it's ignored by the loss
            T5labels = torch.tensor(T5labels).to(device)
            T5labels[T5labels == T5tokenizer.pad_token_id] = -100
            if code_prof:
                print(f"T5 input processing time: {data_time.val}")

            label_test = data['label'].to(device)
            # print(data['label'].detach().cpu().numpy())
            true_label_list+=list(data['label'].detach().cpu().numpy())
            model.zero_grad()
            data_time.reset()
            model_start = time.time()
            
            vencdec_out_test, main_out_test, deBERTa_loss_test = model(vit_inputs, deBERTainputs, deBERTalabels, T5input_ids, T5attention_mask, T5labels)
            data_time.update(time.time() - model_start)
            if code_prof:
                print(f"model processing time: {data_time.val}")
            
            # print(main_out_val.squeeze())
            vencdec_loss_test = vencdec_out_test.loss
            
            try:
                main_loss_test = criterion(main_out_test.squeeze(), label_test)
            except:
                print(main_out_test.squeeze)
                print(label_test)
                main_loss_test = criterion(main_out_test, label_test)
            # print(main_loss_val)
            loss_test = 0.5*vencdec_loss_test+0.3*main_loss_test+0.2*deBERTa_loss_test
            # print(f"val_vencdec_loss: {vencdec_loss_val.item()} | val_main_loss: {main_loss_val.item() | Total VAL loss: {loss_val.item()}}")

            _, predicted_test = torch.max(main_out_test.data, 1)
            predicted_test_cur = list(predicted_test.detach().cpu().numpy())
            predicted_label_list+=list(predicted_test_cur)
            # print(label_test.size(0))
            total_test += label_test.size(0)
            correct_test += (predicted_test == label_test).sum().item()                
            total_vencdec_loss_test += vencdec_loss_test.item()
            total_main_loss_test += main_loss_test.item()
            total_deBERTa_loss_test += deBERTa_loss_test.item()
            total_loss_test += loss_test.item()
            
                        
            output_sequences = model.model_T5.generate(input_ids=T5input_ids,attention_mask=T5attention_mask,do_sample=False, min_length= 0, max_length=512)
            generated_text = T5tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            generated_result+=generated_text
            # break
   

    test_acc = 100 * correct_test / total_test
    test_loss = total_loss_test/total_test
    test_vencdec_loss = total_vencdec_loss_test/total_test
    test_main_loss = total_main_loss_test/total_test
    test_deBERTa_loss = total_deBERTa_loss_test/total_test
    
    return  test_acc, test_loss, test_vencdec_loss, test_main_loss, test_deBERTa_loss, generated_result, true_label_list, predicted_label_list, img_list


# In[ ]:


mode = ''
for i in range(2):
    test_df = pd.read_csv(test_path, index_col=0)
    if i == 1:
        mode = '_bestckp'  
        try:
            del model
        except:
            pass
        path = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')        
        n_out=3
        model = MM(n_out)
        model.load_state_dict(torch.load(path))
        model.to(device)
    else:
        mode = ''  
        try:
            del model
        except:
            pass
        path = os.path.join(exp_path, "final.pt")
        n_out=3
        model = MM(n_out)
        model.load_state_dict(torch.load(path))
        model.to(device)
        
    test_acc, test_loss, test_vencdec_loss, test_main_loss, test_deBERTa_loss, generated_result, true_label_list, predicted_label_list, img_list = test_model(model)
    if i==0:
        print('---Last checkpoint results---')
    else:
        print('---Best checkpoint results---')
    print("test_acc, test_loss, test_vencdec_loss, test_main_loss, test_deBERTa_loss")
    print(test_acc, test_loss, test_vencdec_loss, test_main_loss, test_deBERTa_loss)
    print(classification_report(true_label_list, predicted_label_list, target_names=['Hero', 'Victim', 'Villain']))
    print(f"generated sequences: {len(generated_result)}")
    # generated_result
    
    resdf = pd.DataFrame.from_dict({'images': img_list, 'generated_result': generated_result})
    sid_list = test_df["siddhant's explanations"].tolist()
    tharun_list = test_df["tharun's explanations"].tolist()
    resdf['sid_exp'] = sid_list
    resdf['tharun_exp'] = tharun_list
    resdf.to_csv(os.path.join(exp_path, exp_name+'_GenTrue_exp'+mode+'.csv'), index=False)
    with open(os.path.join(exp_path, 'hyp'+mode+'.txt'), 'w+') as hf:
        hf.write('\n'.join(generated_result))

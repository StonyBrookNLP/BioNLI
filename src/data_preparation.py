# %%
import torch
import pickle
import pandas as pd
import os
from collections import Counter
from transformers import AutoTokenizer
import random
import numpy as np
import pickle
import random

# %%
base_path='entailment_data/v10'
base_save='saved_model/v10_2'
# suffix='_onlyconc' # activate for only concl
suffix='' 
seq_length=512
print('suffix is:',suffix)

dataset_df=pd.read_csv(os.path.join(base_path,'entailment_data.csv'),index_col=False)


# %%
model_name ="michiyasunaga/BioLinkBERT-base"
# model_name= "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
print('used model is: ',model_name)
print('Loading BERT tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
special_tokens = [ '<exp>',  # 32101
                    '<re>',  # 32102
                    '<er>',  # 32103
                    '<el>',  # 32104
                    '<le>',  # 32105
                    '<end>'
                    ]

special_tokens_dict = {'additional_special_tokens': special_tokens}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

# %%
def sentence_encoder(sentence1,sentence2,labels_map,labels=None,seq_length=128,tokenizer=tokenizer, lowercase=False):
    input_ids = []
    attention_masks = []
    token_types=[]
    # For every sentence...
    for i,sent1 in enumerate(sentence1):
        sent2=None
        if not sentence2 is None:
            sent2=sentence2[i]
            if lowercase:
                sent2 = sent2.lower()

        if lowercase:
            sent1 = sent1.lower()
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            text=sent1,                      # Sentence to encode.
                            text_pair=sent2,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = seq_length,           # Pad & truncate all sentences.
                            truncation=True,
                            truncation_strategy='only_first',
                            padding='max_length',
                            # pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            return_token_type_ids=True,
                       )
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        token_types.append(encoded_dict['token_type_ids'])

    # Convert the lists into tensors.
    
    input_ids = torch.cat(input_ids, dim=0)
    print(input_ids.shape)
    exit()
    attention_masks = torch.cat(attention_masks, dim=0)
    token_types =  torch.cat(token_types, dim=0)
    
    try:
        labels_out=[labels_map[c] for c in labels]
        labels = torch.tensor(labels_out)
    except Exception as e:
        print('error occured:%s'%str(e))
        pass
    return(input_ids,attention_masks,token_types,labels)


# %%
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
print(list(dataset_df))
ids =dataset_df['index'].values

# %%
import random

def dataset_labeling(x,p1=0.9,p2=0.7,p3=0.4,p4=0.1):
    r=x['prob']
    if x['label_cat']=='generation_nd_SEN' \
        or x['label_cat']=='generation_nd_SRE' \
            or x['label_cat']=='generation' \
                or x['label_cat']=='generation_nd' \
                    or x['label_cat']=='SRE':
        if r<p1:
            return 'train' 
        else:
            return 'test'
    if x['label_cat']=='pos':
        if r < p2:
            return 'train'
    if x['label_cat']=='SEP' \
            or x['label_cat']=='SEN' \
                or x['label_cat']=='posToNeg' \
                        or x['label_cat']=='negToPos':
        if r < p4:
            return 'train'

    if x['label_cat']=='LPR' \
        or x['label_cat']=='swap_number':
        if r < p3:
            return 'train'
    if x['label_cat']=='SREO':
        if r< p4:
            return 'train'

def select_row(rows):
    ind=0
    ## return any item in this categories
    for _, x in rows.iterrows():
        if x['label_cat'] in ['generation_nd_SEN' ,'generation_nd_SRE' ,'generation','generation_nd' ,'SRE']:
            return rows.iloc[[ind]]
        ind +=1
    ## randomly select one category for the other data to avoid duplication in training set
    return rows.iloc[[random.randint(0,len(rows)-1)]]

def call_encoder(sents,sec_sents,labels,suffix,labels_map):
    if suffix=='':
        input_ids,attention_masks,token_types,lbl_tns = \
        sentence_encoder(sents,sec_sents,labels_map,labels,seq_length,lowercase=False)

    else:
        input_ids,attention_masks,token_types,lbl_tns = \
        sentence_encoder(sec_sents,None,labels_map,labels,seq_length,lowercase=False)

    dataset = TensorDataset(input_ids,token_types, attention_masks, lbl_tns.long())
    return(dataset)

def call_saver(dataset,df,data='train',tag='blnc'):
    with open(os.path.join(base_save,'%s_dataset_%s_%s%s'%(data,tag,model_name[:5],suffix)),'wb') as f:
        pickle.dump(dataset,f)

    with open(os.path.join(base_save,'%s_df_%s_%s%s.csv'%(data,tag,model_name[:5],suffix)),'w') as f:
        df.to_csv(f,index=False)

# %%
from torch.utils.data import TensorDataset
try:
    with open(os.path.join(base_save,'train_dataset_%s%s'%(model_name[:5],suffix)),'rb') as f:
        train_dataset=pickle.load(f)
    with open(os.path.join(base_save,'val_dataset_%s%s'%(model_name[:5],suffix)),'rb') as f:
        val_dataset=pickle.load(f)
    with open(os.path.join(base_save,'train_df_%s%s.csv'%(model_name[:5],suffix)),'r') as f:
        train_df=pd.read_csv(f)
    with open(os.path.join(base_save,'val_df_%s%s.csv'%(model_name[:5],suffix)),'r') as f:
        val_df=pd.read_csv(f)
    print('input data loaded')
except Exception as e:
    print('data can not be loaded, creating dataset instances. The loading error is:',str(e))
    dataset_df=dataset_df[dataset_df['label_cat']!='SET']
    dataset_df['prob']=dataset_df.apply(lambda x:random.random(),axis=1)
    print('label dist in all data: ',Counter(dataset_df['label_cat'].values))
    labels_map={'pos':1}
    for item in set(dataset_df['label_cat'].values):
        if item=='pos':
            continue
        labels_map[item]=0

    print('label map is: ',labels_map)

    ## create balance dataset
    dataset_df['dataset']=dataset_df.apply(lambda x: dataset_labeling(x),axis=1)
    val_df_blnc=dataset_df[dataset_df['dataset']=='test']
    val_ids_blnc=set(val_df_blnc['pmid'])
    train_df_blnc=dataset_df[~dataset_df['pmid'].isin(val_ids_blnc)]
    train_df_blnc=train_df_blnc.sample(frac=1).groupby('pmid', as_index=False).apply(lambda x: select_row(x)).dropna()
    
    print('Number of positive instances in train blnc: ',len(train_df_blnc[train_df_blnc['label_cat']=='pos']))
    print('Number of negative instances in train blnc: ',len(train_df_blnc[train_df_blnc['label_cat']!='pos']))
    train_set_blnc= train_df_blnc['supp_set'].values
    train_sec_sents_blnc= train_df_blnc['conclusion'].values
    train_labels_blnc= train_df_blnc['label_cat'].values
    print('training blnc size:',len(train_df_blnc))
    print('label dist in training blnc: ',Counter(train_labels_blnc))
    train_ids_blnc=set(train_df_blnc['pmid'])


    val_df_blnc=dataset_df[~dataset_df['pmid'].isin(train_ids_blnc)]
    print('Number of positive instances in dev blnc: ',len(val_df_blnc[val_df_blnc['label_cat']=='pos']))
    print('Number of negative instances in dev blnc: ',len(val_df_blnc[val_df_blnc['label_cat']!='pos']))
    val_set_blnc= val_df_blnc['supp_set'].values
    val_sec_sents_blnc= val_df_blnc['conclusion'].values          
    val_labels_blnc= val_df_blnc['label_cat'].values
    print('validation size blnc:',len(val_df_blnc))
    print('label dist in validation blnc: ',Counter(val_labels_blnc))





    ## create as large as possible dataset
    dataset_df_sampled=dataset_df.sample(frac=1)
    dataset_df_sampled['dataset']=dataset_df_sampled.apply(lambda x: dataset_labeling(x,p1=0.9,p2=0.65,p3=0.3,p4=0.2),axis=1)
    val_df_lrg=dataset_df_sampled[dataset_df_sampled['dataset']=='test']
    val_ids_lrg=set(val_df_lrg['pmid'])
    train_df_lrg=dataset_df_sampled[~dataset_df_sampled['pmid'].isin(val_ids_lrg)]
    train_df_lrg=train_df_lrg.dropna(subset=['dataset'])
    train_df_lrg=train_df_lrg.sample(frac=1).groupby('pmid', as_index=False).apply(lambda x: select_row(x)).dropna()
    
    print('Number of positive instances in train lrg: ',len(train_df_lrg[train_df_lrg['label_cat']=='pos']))
    print('Number of negative instances in train lrg: ',len(train_df_lrg[train_df_lrg['label_cat']!='pos']))
    train_set_lrg= train_df_lrg['supp_set'].values
    train_sec_sents_lrg= train_df_lrg['conclusion'].values
    train_labels_lrg= train_df_lrg['label_cat'].values
    print('training size lrg:',len(train_df_lrg))
    print('label dist in training lrg : ',Counter(train_labels_lrg))
    train_ids_lrg=set(train_df_lrg['pmid'])


    val_df_lrg=dataset_df_sampled[~dataset_df_sampled['pmid'].isin(train_ids_lrg)]
    print('Number of positive instances in dev lrg: ',len(val_df_lrg[val_df_lrg['label_cat']=='pos']))
    print('Number of negative instances in dev lrg: ',len(val_df_lrg[val_df_lrg['label_cat']!='pos']))
    val_set_lrg= val_df_lrg['supp_set'].values
    val_sec_sents_lrg= val_df_lrg['conclusion'].values          
    val_labels_lrg= val_df_lrg['label_cat'].values
    print('validation size lrg:',len(val_df_lrg))
    print('label dist in validation lrg: ',Counter(val_labels_lrg))


# %%
    train_dataset_blnc=call_encoder(train_set_blnc,train_sec_sents_blnc,train_labels_blnc,suffix,labels_map)
    train_dataset_lrg=call_encoder(train_set_lrg,train_sec_sents_lrg,train_labels_lrg,suffix,labels_map)
    val_dataset_blnc=call_encoder(val_set_blnc,val_sec_sents_blnc,val_labels_blnc,suffix,labels_map)
    val_dataset_lrg=call_encoder(val_set_lrg,val_sec_sents_lrg,val_labels_lrg,suffix,labels_map)
    call_saver(train_dataset_blnc,train_df_blnc,data='train',tag='blnc')
    call_saver(val_dataset_blnc,val_df_blnc,data='val',tag='blnc')
    call_saver(train_dataset_lrg,train_df_lrg,data='train',tag='lrg')
    call_saver(val_dataset_lrg,val_df_lrg,data='val',tag='lrg')

# %%

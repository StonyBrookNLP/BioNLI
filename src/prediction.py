# Classifier_2S
# ## Setup

# %%
## How to run this code:
# python Prediction_2S.py --model_name "michiyasunaga/BioLinkBERT-base"  --val_batch_size 256 --suffix "_onlyconc" --tag 'lrg' --device 'cuda:0'

# %%

import torch
import pickle
import pandas as pd
import os
from collections import Counter
import argparse
from torch.utils.data import TensorDataset
# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", help="pretrained model name") #"michiyasunaga/BioLinkBERT-base"
parser.add_argument("--val_batch_size", type=int, default=16)
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--base_data",type=str, default="saved_model/v10_2", help="")
parser.add_argument("--base_save",type=str, default="saved_model/v10_2",help="")
parser.add_argument("--suffix",type=str, default="",help="")
parser.add_argument("--tag",type=str, default="blnc",help="lrg or blnc")
parser.add_argument("--device",type=str, default="cuda:0",help="cuda")
args = parser.parse_args("")

# %%

#CUDA_LAUNCH_BLOCKING=1
# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device(args.device)
    

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
    

# %%


# base_path='entailment_data/v9'
base_save=args.base_save
base_data=args.base_data


suffix=args.suffix
print('suffix is:',suffix)
val_batch_size = args.val_batch_size
tag=args.tag
print('tag is:',tag)

# %%
from transformers import AutoTokenizer
import numpy as np
model_base =args.model_name
print('model base is: ',model_base )
# model_name=os.path.join(base_save,'%s_2sent_class_layer%s_top3'%(model_base[:5],suffix))
model_name=os.path.join(base_save,'%s_2sent_class_layer_%s%s_top3'%(model_base[:5],tag,suffix))
# Load the BERT tokenizer.
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
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



# %%

with open(os.path.join(base_save,'val_dataset_%s_%s%s'%(tag,model_base[:5],suffix)),'rb') as f:
    val_dataset=pickle.load(f)
with open(os.path.join(base_save,'val_df_%s_%s%s.csv'%(tag,model_base[:5],suffix)),'r') as f:
    val_df=pd.read_csv(f)
    val_labels_df= val_df['label_cat'].values

with open('saved_model/v10_2/train_df_lrg_%s%s.csv'%(model_base[:5],suffix),'r') as f:
    pre_train_df=pd.read_csv(f)
pre_train_ids=list(pre_train_df['pmid'])

# %%
first_elements=[]
second_elements=[]
third_elements=[]
forth_elements=[]
for i,item in val_df.iterrows():
    if item['pmid'] not in pre_train_ids:
        first_elements.append(val_dataset[i][0])
        second_elements.append(val_dataset[i][1])
        third_elements.append(val_dataset[i][2])
        forth_elements.append(val_dataset[i][3])

new_val_df=val_df[~val_df['pmid'].isin(pre_train_ids)]
tns_first=torch.stack(first_elements, dim=0)
tns_second=torch.stack(second_elements, dim=0)
tns_third=torch.stack(third_elements, dim=0)
tns_forth=torch.tensor(forth_elements)
new_val_dataset = TensorDataset(tns_first,tns_second, tns_third, tns_forth.long())
print('new dataset has length of:',len(new_val_dataset))
print('new dataset has unique number of:',len(set(new_val_df['pmid'])))
new_val_labels_df= new_val_df['label_cat'].values
print('label dist in new validation blnc: ',Counter(new_val_labels_df))
# exit()
# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification



# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.


# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            new_val_dataset, # The validation samples.
            sampler = SequentialSampler(new_val_dataset), # Pull out batches sequentially.
            batch_size = val_batch_size # Evaluate with this batch size.
        )

# %%
from sklearn.metrics import accuracy_score



model = AutoModelForSequenceClassification.from_pretrained(model_name,
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states)
)
model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
model.to(device)


# %%
import random
import numpy as np
import sklearn

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128



# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
val_predictions = []

# Measure the total training time for the whole run.
total_t0 = time.time()
last_best = 0
for name, param in model.named_parameters():
    param.requires_grad = False
# %%
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
def myeval(validation_dataloader, device, model):
    t0 = time.time()
    preds = np.array([])
    true_labels = np.array([])
    model.eval()
    for batch in validation_dataloader:

      
            b_input_ids = batch[0].to(device)
            b_token_types=batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)
            
            


            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids, 
                            token_type_ids=b_token_types, 
                            attention_mask=b_input_mask,
                            return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the 
            # output values prior to applying an activation function like the softmax.
            logits = result.logits
            # Accumulate the validation loss.

            # Move logits and labels to CPU
            logits_c = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            preds = np.append(preds,np.argmax(logits_c,1))
            true_labels = np.append(true_labels,label_ids)
            


    # Calculate the average loss over all of the batches.
        
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    print(classification_report(y_pred=list(preds),y_true=list(true_labels),labels=[0,1]))  
    print(confusion_matrix(y_pred=list(preds),y_true=list(true_labels),labels=[0,1]))
    acc = accuracy_score(y_true=list(true_labels),y_pred=list(preds))
    return(list(preds),acc)   





preds,acc= myeval(validation_dataloader, device, model)   
new_val_df['predictions_%s%s_%s'%(tag,model_base[:5],suffix)]= preds
new_val_df.to_csv('predictions_%s%s_%s.csv'%(tag,model_base[:5],suffix),index=False)
print("")
print("Validation complete!")
outfile=open(os.path.join(base_save,'dev_results_%s%s_%s.txt'%(tag,model_base[:5],suffix)),'w')

print("Total validation took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
print("Validation accuracy: {:.2f}".format(acc) ,file=outfile)

print('results of the best model on dev set: ',file=outfile)
result_map={} ## a map for per class performance, label_category:[#incorrect,#correct]
for i,j in zip(new_val_labels_df,preds):
    if i=='pos':
        if 'pos' in result_map:
            if j==0:
                result_map['pos'][0] = result_map['pos'][0]+1
            else:
                result_map['pos'][1] = result_map['pos'][1]+1
        else:
            if j==0:
                result_map['pos']= [1,0]
            else:
                result_map['pos']=[0,1]
    else:
        if i in result_map:
            if j==0:
                result_map[i][1] = result_map[i][1]+1
            else:
                result_map[i][0] = result_map[i][0]+1
        else:
            if j==0:
                result_map[i]= [0,1]
            else:
                result_map[i]=[1,0]
weighted_sum=0
all_counts=0
for item in result_map:
    incorrect=result_map[item][0]
    correct=result_map[item][1]
    weighted_sum += correct
    all_counts += (correct+incorrect)
    print(item,  "{:.3f}".format(correct/(correct+incorrect)),correct+incorrect,file=outfile)
print("overall accuracy: {:.3f}".format(weighted_sum/(all_counts)),all_counts,file=outfile)
print("negative accuracy: {:.3f}".format((weighted_sum-result_map['pos'][1])/(all_counts-result_map['pos'][0]-result_map['pos'][1])),\
    all_counts-result_map['pos'][0]-result_map['pos'][1],file=outfile)

print('suffix: ',suffix,file=outfile)


# %%
### evaluate a negative instance as correctly classified if all 4 versions are correctly classified
print('combined results',file=outfile)
performance=[]
val_ids=set(list(new_val_df['index']))
print('length of the validation set:', len(val_ids),file=outfile)
overal_TP=0
overal_TN=0
overal_FP=0
overal_FN=0
labels_ture =[]
labels_pred =[]
allts=[]
for ind in val_ids:
    true_pos=0
    true_neg=0
    all_trues=0
    row=new_val_df[new_val_df['index']==ind]
    true_labels=list(row['label_cat'])
    pred_labels=list(row['predictions_%s%s_%s'%(tag,model_base[:5],suffix)])
    for tl, pl in zip(true_labels,pred_labels):
        if tl=='pos' and pl==1.0:
            true_pos +=1
            all_trues+=1
        elif  tl!='pos' and pl==0.0:
            true_neg +=1
            all_trues +=1
    prf=max(true_neg,true_pos)/len(pred_labels)
    allt=all_trues/len(pred_labels)
    allts.append(allt)
    if tl=='pos' and prf==1: ## true: pos, all predicted: pos
        overal_TP+=1
        labels_ture.append(1)
        labels_pred.append(1)
    elif tl!='pos' and prf==1: ## true: neg, all predicted: neg
        overal_TN+=1
        labels_ture.append(0)
        labels_pred.append(0)
    elif tl=='pos' and prf!=1: ## true: pos, all predicted: neg
        overal_FN+=1
        labels_ture.append(1)
        labels_pred.append(0)
    elif tl!='pos' and prf!=1: ## true: neg, all predicted: pos
        overal_FP+=1
        labels_ture.append(0)
        labels_pred.append(1)
    performance.append(prf)

outfile.close()
# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# plt.grid(True)
plt.title('Perturbations Classified Correctly in Dataset')
plt.xlabel('Percentage of Correctly Classified Perturbations')
plt.ylabel('Likelihood of Occurrence')
n, bins, patches = plt.hist(performance, 4,weights=np.ones(len(performance)) / len(performance), density=False, histtype='bar', cumulative=True, color='b',edgecolor='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

# %%
#### new plots
c_performance={}
for item in performance:
    c_performance[round(item, 2)]=round(performance.count(item)/float(len(performance)),4)
sorted_performance = dict(sorted(c_performance.items()))
keys = list(sorted_performance.keys())
values = list(sorted_performance.values())
new_keys_values={0.25:sum(values[:3]),0.5:sum(values[:7]),0.62:sum(values[:11]),0.75:sum(values[:13]),0.83:sum(values[:16]),0.89:sum(values) }
df = pd.DataFrame({'keys':list(new_keys_values.keys()), 'val':list(new_keys_values.values())})

df=df.round(3)
df_plot = df.plot.bar(x='keys', y='val', width=1,rot=0,color='blue',edgecolor='black',title='Percentage of Perturbations Classified Correctly',ylabel='Dev Size', xlabel='Percentage of Correctly Classified Perturbations',legend=False)
ax=df_plot.get_figure()


df_plot = df.plot(kind='bar',x='val', y='keys', width=1,rot=0,color='blue',edgecolor='black',\
    title='Percentage of Perturbations Classified Correctly',\
        ylabel='Dev Size', \
            xlabel='Percentage of Correctly Classified Perturbations',\
                legend=False, grid=True,\
                    yticks=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],\
                       use_index=True)
ax=df_plot.get_figure()
df_plot.set_xticklabels([0.1,0.2,0.4])


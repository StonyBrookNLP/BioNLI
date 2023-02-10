
# ## Setup

# %%
import torch
import pickle
import pandas as pd
import os
from collections import Counter
from transformers import AutoTokenizer,AutoModel
import random
import numpy as np
import pickle
import argparse

  

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", help="pretrained model name") #"michiyasunaga/BioLinkBERT-base"
parser.add_argument("--epochs",  type=int,default=10, help="")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--val_batch_size", type=int, default=16)
parser.add_argument("--seq_length", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--base_path",type=str, default="entailment_data/v10", help="")
parser.add_argument("--base_save",type=str, default="saved_model/v10_2",help="")
parser.add_argument("--suffix",type=str, default="",help="")
parser.add_argument("--tag",type=str, default="lrg",help="lrg or blnc")
parser.add_argument("--device",type=str, default="cuda:0",help="cuda")
args = parser.parse_args()


# %%
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
epochs =args.epochs
base_path=args.base_path
base_save=args.base_save


suffix=args.suffix # activate for only concl
print('suffix is:',suffix)
batch_size=args.val_batch_size
val_batch_size =args.val_batch_size 
seq_length = args.seq_length
LR=args.learning_rate
# tag='blnc','lrg'
tag=args.tag
print('tag is:',tag)

dataset_df=pd.read_csv(os.path.join(base_path,'entailment_data.csv'),index_col=False)


# %%
model_name= args.model_name
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


def save(model,tokenizer):
    # save
    model.save_pretrained(os.path.join(base_save,'%s_2sent_class_layer_%s%s_top3'%(model_name[:5],tag,suffix)))
    tokenizer.save_pretrained(os.path.join(base_save,'%s_2sent_class_layer_%s%s_top3'%(model_name[:5],tag,suffix)))



# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AdamW

try:
   
    with open(os.path.join(base_save,'train_dataset_%s_%s%s'%(tag,model_name[:5],suffix)),'rb') as f:
        train_dataset=pickle.load(f)
    with open(os.path.join(base_save,'val_dataset_%s_%s%s'%(tag,model_name[:5],suffix)),'rb') as f:
        val_dataset=pickle.load(f)
    with open(os.path.join(base_save,'train_df_%s_%s%s.csv'%(tag,model_name[:5],suffix)),'r') as f:
        train_df=pd.read_csv(f)
    with open(os.path.join(base_save,'val_df_%s_%s%s.csv'%(tag,model_name[:5],suffix)),'r') as f:
        val_df=pd.read_csv(f)
    print('input data loaded')
except Exception as e:
    print('error in loading the dataset:',str(e))
    print('please rerun datapreparation_2S.py')
    exit()



## order of items in the dataset: input_ids, token_ids, attention_mask, label
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )



# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = val_batch_size # Evaluate with this batch size.
        )

# %%
from sklearn.metrics import accuracy_score



model = AutoModelForSequenceClassification.from_pretrained(model_name, #AutoModelForSequenceClassification.from_pretrained(model_name,
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states)
)
model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
model.to(device)
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = LR,#2e-5,#4e-5,#lr = 4e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 25, # Default value in run_glue.py
                                            num_training_steps = total_steps)



# %%
import random
import numpy as np


# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []
val_predictions = []

# Measure the total training time for the whole run.
total_t0 = time.time()
last_best = 0
for name, param in model.named_parameters():
        if 'classifier' not in name and 'cls' not in name and \
        'layer.11' not in name and 'layer.10' not in name and 'layer.9' not in name: #\
        #         and 'layer.8' not in name and 'layer.7' not in name \
        #          and 'layer.6' not in name  and 'layer.5' not in name \
        #   and 'layer.4' not in name and 'layer.3' not in name: # classifier layer
                    param.requires_grad = False
# %%
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
def myeval(validation_dataloader, device, model):
    t0 = time.time()
    preds = np.array([])
    true_labels = np.array([])
    total_eval_loss=0
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
)
    print(classification_report(y_pred=list(preds),y_true=list(true_labels),labels=[0,1]))  
    print(confusion_matrix(y_pred=list(preds),y_true=list(true_labels),labels=[0,1]))
    acc = accuracy_score(y_true=list(true_labels),y_pred=list(preds))
    macro_f1=f1_score(y_pred=list(preds),y_true=list(true_labels),labels=[0,1],average='macro')
    print('macro F1 is {:.2f}'.format(macro_f1))
    return(acc,macro_f1)   

# %%
from torch import nn
classifier=nn.Linear(768, 2).to(device)
criterion = nn.CrossEntropyLoss().to(device)
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0


    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()
    

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_token_types=batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        model.zero_grad()        

        result = model(b_input_ids, 
                       token_type_ids=b_token_types, 
                       attention_mask=b_input_mask, 
                       labels=b_labels,
                       return_dict=True)
        loss = result.loss
        logits = result.logits

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()
        optimizer.zero_grad()
        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()
        
        

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    acc,macro_f1= myeval(validation_dataloader, device, model)
    if macro_f1 > last_best:
        last_best = macro_f1
        save(model,tokenizer)
        print('new macro f1 achieved: %3f'%macro_f1)



     
print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
print('Best macro f1 achieved: %3f'%last_best)
print('suffix is:',suffix)




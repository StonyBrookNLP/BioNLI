import random
import pandas as pd
import re
import json
import spacy

train_df=pd.read_csv('data/v8_1/used_inputs_for_train_v8.csv')
entity_df=pd.DataFrame()
def save_entities(sent,file):
    entity_type_text_map={}
    try:
        spacy.prefer_gpu()
        nlp = spacy.load("en_ner_bionlp13cg_md")
        doc=nlp(sent)
        for ent in doc.ents:
            entity_type_text_map[ent.text]=ent.label_
    except Exception as e:
        print('error occured: ',e)
    with open(file,'a') as f:
        json.dump(entity_type_text_map,f)
        f.write('\n')
    return entity_type_text_map
        

train_exp=list(train_df['conclusion'])
train_supp=list(train_df['potential_supp'])
pmids=list(train_df['pmid'])
index=-1
for line,supp_set,pmid in zip(train_exp,train_supp,pmids):
    index+=1
    if index % 1000 ==0:
        print('processed %d items'%index)
    conc_ent=save_entities(line,'conclusion_entity_types_v2.json')
    supp_ent=save_entities(supp_set,'supp_set_entity_types_v2.json')
    new_row={'pmid':pmid,'concl_ent':conc_ent,'supp_ent':supp_ent}
    entity_df=entity_df.append(new_row,ignore_index=True)
    entity_df.to_csv('data/v8_1/train_entity_types.csv',index=False)
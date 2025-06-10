
## ChemBERTA


from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250")
model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250")        

tokenizer_2 = AutoTokenizer.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
model_2 = AutoModel.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")

MODELS = {'bpe': model,
          'tok': model_2}
TOKENIZERS = {'bpe': tokenizer,
              'tok': tokenizer_2}
    
def get_chemberta_encoding(smis, version='tok'):
    '''
    smis: single SMILES string OR a list of SMILES.
    '''
    tokenizer = TOKENIZERS[version]
    model = MODELS[version]

    
# # # # # # # # # #
#     encoded_input = tokenizer(smis, padding=True, truncation=True, max_length=128, return_tensors='pt')
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     rep = mean_pooling(model_output, encoded_input['attention_mask'])

# # # # # # # # # #
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     device = "cuda"
#     model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(device)
    
# model = nn.DataParallel(model, device_ids=use_cuda)
    
    encoded_input = tokenizer(smis, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        
#     model_output = model_output.cpu().detach().numpy()
#     encoded_input = encoded_input.cpu().detach().numpy()
    rep = mean_pooling(model_output, encoded_input['attention_mask'])
    rep = rep.cpu().detach().numpy()

    return rep







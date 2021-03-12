import torch
from transformers import RobertaTokenizerFast, DistilBertTokenizerFast
import json
from transformers import DistilBertModel, BertForSequenceClassification, BertTokenizer

import os
import tarfile
import io
import base64
import json
import re
import pdb 

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# path to all the files that will be used for inference
path = f"./app/models/"

# self.model_path = self.path + "traced_bert_epoch_1.pt"
model_path = path + "custom_trained_model.bin"

#tokenizer_path = "./app/models/bert-large-portuguese-cased"
tokenizer_path = "neuralmind/bert-large-portuguese-cased"

# self.model = torch.jit.load(self.model_path)
model = BertForSequenceClassification.from_pretrained(
    tokenizer_path, num_labels = 17,
    #local_files_only=True
)
model.load_state_dict(
    torch.load(model_path, map_location=device)
)
                                    
tokenizer = BertTokenizer.from_pretrained(
    tokenizer_path, do_lower_case=True, torchscript=True, 
)

LABELS = {
   0:"A",
   1:"B",
   2:"C"
}

class ClassificationProcessor:
    def __init__(self):
        
        self.model = model
        self.model.eval()
        
        self.tokenizer = tokenizer

        self.labels = LABELS


    def tokenize(self, input_text: str, query: str = None):
        """
        Method to tokenize the textual input
        :param input_text: Input text
        :param query: Query in case of Question Answering service.
        :return: Returns encoded text for inference
        """        
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=20, 
            padding=True,
            return_tensors='pt'
        )

        inputs.to(device)

        return inputs

    def lookup(self, pred: int):
        """
        Function to perform look up against the mapping json file. Only applicable for classificaiton and sentiment analysis.
        :return: Correct category for the prediction.
        """
        #return self.config[str(int(pred.item()))]
        return self.labels[pred]

    def inference(self, input_text: str, query: str = None):
        """
        Method to perform the inference
        :param input_text: Input text for the inference
        :param query: Input qwuery in case of QnA
        :return: correct category and confidence for that category
        """
        tokenized_inputs = self.tokenize(input_text)
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        out = outputs[0]
        confidence, pred = torch.max(out, dim=1)
        sentiment_class = self.lookup(int(pred))

        return sentiment_class, float(confidence) * 10

import json
import os
from tqdm import tqdm
from typing import List
from IPython.core.debugger import set_trace
from pprint import pprint
from transformers import AutoModel, BertTokenizerFast
import torch
from torch.utils.data import DataLoader, Dataset
from common.utils import Preprocessor
from .tplinker import (HandshakingTaggingScheme,
                      DataMaker4Bert, 
                      TPLinkerBert)
from .config import eval_config

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)



class RelationExtractor:
    def __init__(self, config, model_path, rel2id_path):
        self.config = config
        self.hyper_parameters = config["hyper_parameters"]

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.deterministic = True

        rel2id = json.load(open(rel2id_path, "r", encoding = "utf-8"))
        self.handshaking_tagger = HandshakingTaggingScheme(rel2id = rel2id, max_seq_len = self.hyper_parameters["max_test_seq_len"])

        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
        self.data_maker = DataMaker4Bert(tokenizer, self.handshaking_tagger)
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens = False)["offset_mapping"]

        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=False, do_lower_case=False)
        tokenize = tokenizer.tokenize
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens=False)["offset_mapping"]
        self.preprocessor = Preprocessor(tokenize_func=tokenize, get_tok2char_span_map_func=get_tok2char_span_map)

        roberta = AutoModel.from_pretrained(config["bert_path"])
        self.model = TPLinkerBert(roberta, 
                                len(rel2id), 
                                self.hyper_parameters["shaking_type"],
                                self.hyper_parameters["inner_enc_type"],
                                self.hyper_parameters["dist_emb_size"],
                                self.hyper_parameters["ent_add_dist"],
                                self.hyper_parameters["rel_add_dist"],
                                ).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def __call__(self, texts: List[str]):
        '''
        test_data: if split, it would be samples with subtext
        ori_test_data: the original data has not been split, used to get original text here
        '''
        data = [{'text': text for text in texts}]
        indexed_test_data = self.data_maker.get_indexed_data(data, self.hyper_parameters['max_test_seq_len'], data_type = "test")
        test_dataloader = DataLoader(MyDataset(indexed_test_data), 
                                batch_size = self.hyper_parameters['batch_size'], 
                                shuffle = False, 
                                num_workers = 6,
                                drop_last = False,
                                collate_fn = lambda data_batch: self.data_maker.generate_batch(data_batch, data_type = "test"),
                                )
        
        pred_sample_list = []
        for batch_test_data in tqdm(test_dataloader, desc = "Predicting"):
            sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, _, _, _ = batch_test_data

            batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(self.device), 
                                    batch_attention_mask.to(self.device), 
                                    batch_token_type_ids.to(self.device))


            with torch.no_grad():
                batch_ent_shaking_outputs, batch_head_rel_shaking_outputs, batch_tail_rel_shaking_outputs = self.model(batch_input_ids, 
                                                        batch_attention_mask, 
                                                        batch_token_type_ids, 
                                                        )
   
            batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = torch.argmax(batch_ent_shaking_outputs, dim = -1), torch.argmax(batch_head_rel_shaking_outputs, dim = -1), torch.argmax(batch_tail_rel_shaking_outputs, dim = -1)

            for index, sample in enumerate(sample_list):
                text = sample["text"]
                tok2char_span = tok2char_span_list[index]
                ent_shaking_tag, head_rel_shaking_tag, tail_rel_shaking_tag = batch_ent_shaking_tag[index], batch_head_rel_shaking_tag[index], batch_tail_rel_shaking_tag[index]
                tok_offset, char_offset = 0, 0
                rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                                        ent_shaking_tag, 
                                                                        head_rel_shaking_tag, 
                                                                        tail_rel_shaking_tag, 
                                                                        tok2char_span, 
                                                                        tok_offset = tok_offset, char_offset = char_offset)
                pred_sample_list.append({
                    "text": text,
                    "relation_list": rel_list,
                })

        return pred_sample_list



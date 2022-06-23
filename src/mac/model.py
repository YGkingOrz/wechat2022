import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,RobertaModel,RobertaConfig,BertConfig,BertForMaskedLM,AutoModel
import gc
from category_id_map import CATEGORY_ID_LIST
import os
from transformers.models.bert.modeling_bert import BertEmbeddings,BertEncoder
# from qq_model.qq_uni_model import QQUniModel
# from config_qq.data_cfg import *
# from config_qq.model_cfg import *
# from config_qq.pretrain_cfg import *


class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        #两层感知机
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
 
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self):
        super().__init__()
        self.norm= nn.BatchNorm1d(768)
        self.dense = nn.Linear(768, 300)
        self.norm_1= nn.BatchNorm1d(300)
        self.dropout = nn.Dropout(0.3)
        self.out_proj = nn.Linear(300, len(CATEGORY_ID_LIST))

    def forward(self, features, **kwargs):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)        
        x = self.out_proj(x)
        
        return x  

def rm_forward(key_str, j):
    key_list = key_str.split('.')
    key_list = key_list[j:]
    return '.'.join(key_list) 


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        config = BertConfig.from_pretrained(args.bert_dir) 
        
        config.output_hidden_states=True
        self.bert=AutoModel.from_pretrained(args.bert_dir,output_hidden_states=True,cache_dir=None,return_dict=True)
        self.encoder = BertEncoder(config)
        self.embeddings=BertEmbeddings(config)
        
        bert = BertForMaskedLM.from_pretrained(args.bert_dir)
        #net =QQUniModel(MODEL_CONFIG, bert_cfg_dict=BERT_CFG_DICT, model_path=BERT_PATH, task=PRETRAIN_TASK)
        #net.load_state_dict(torch.load("model_state_dict4_1.2219114303588867"))
        #self.bert = net.roberta.bert
        i = 0
        emb_i = 6  # embedding层有6个参数
        enc_i = 198  # encoder层有192个参数
        cls_i = 205  # cls层有7个参数
        # 将bert模型的参数赋给embedding、encoder和cls层
#         for key, value in bert.state_dict().items():
#             if i < emb_i:
#                 key = rm_forward(key, 2)
#                 self.embeddings.state_dict()[key].copy_(value)
#             elif i < enc_i:
#                 key = rm_forward(key, 2)
#                 self.encoder.state_dict()[key].copy_(value)
#             i += 1
        
           
        
        
        
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        self.enhance2 = SENet(channels=768, ratio=args.se_ratio)
        bert_output_size = 768
        self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        #self.fusion2 = ConcatDenseSE(args.fc_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))
        self.fc= nn.Linear(768,len(CATEGORY_ID_LIST))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        self.net = ClassificationHead()
        self.vision_fc=nn.Linear(768,768)
        self.concat_fc=nn.Linear(32+384,32)
        self.attention_fc=nn.Linear(32+384,768)
        self.encoder_fc=nn.Linear(768*4,768)
        self.is5zhe = args.is5zhe
    def forward(self, inputs, inference=False):
        device = torch.device('cuda:0')
        embedding_text= self.bert.embeddings(input_ids=inputs['input_ids']).to(device)        
        video_emb = self.sig(self.vision_fc(inputs['frame_input']))
        embedding_video=self.bert.embeddings(inputs_embeds=video_emb).to(device)
#         embedding_video = self.relu(self.vision_fc(embedding_video))
#         concat_emb = torch.cat((embedding_text,video_emb),dim=1).to(device)
#         attention_video = self.relu(self.concat_fc(concat_emb.transpose(1,2))).transpose(1,2)
        
        embedding = torch.cat((embedding_text,embedding_video),dim=1).to(device)
        mask = torch.cat((inputs['text_mask'],inputs['frame_mask']),dim=1).to(device)
        masks = mask[:,None,None,:]
        masks = (1.0-masks)*-10000.0
        encoder_outputs = self.bert.encoder(embedding,attention_mask=masks,output_hidden_states=True)
        encoder_1 = encoder_outputs['hidden_states'][-1]
        cls_1 = encoder_1[:,0:1,:]
        encoder_2 = encoder_outputs['hidden_states'][-2]
        cls_2 = encoder_2[:,0:1,:]
        encoder_3 = encoder_outputs['hidden_states'][-3]
        cls_3 = encoder_3[:,0:1,:]
        encoder_4 = encoder_outputs['hidden_states'][-4]
        cls_4 = encoder_4[:,0:1,:]
        encoder_cat = torch.cat((encoder_1,encoder_2,encoder_3,encoder_4),dim=1).to(device)
#         encoder_cat = self.relu(self.encoder_fc(encoder_cat))
        encoder_out = torch.mean(encoder_cat,dim=1).to(device)
#         last_encoder = encoder_outputs[-1].to(device)
#         second_encoder = encoder_outputs[-2].to(device)
#         encoder_cat = torch.cat((last_encoder,second_encoder),dim=1).to(device)
        #encoder_outputs=self.enhance2(encoder_outputs)
#         embed_mean=encoder_cat.sum(1)/mask.sum(1).unsqueeze(-1).to(device)
#         x1,x2 = self.enhance2(embed_mean)
        
#         embed_mean = self.enhance2(embed_mean)
#         embed_max=encoder_outputs+(1-mask).unsqueeze(-1)*(-1e10)
#         embed_max=embed_max.max(1)[0].to(device)
#         embed_concat=torch.cat((embed_mean,embed_max),dim=1)
        # print(bert_embedding.shape)
        # fusion_embedding = torch.cat((bert_embedding,inputs['frame_input']),dim=1)
        # text_features=torch.relu(self.dropout(fusion_embedding))
        # text_masks = torch.cat((inputs['title_mask'],inputs['frame_mask']),dim=1)
        # print(text_features.shape)
        # print(text_masks.shape)
        # hidden_states=self.bert(inputs_embeds=text_features,attention_mask=text_masks)[0]
        # print(hidden_states.shape)
        # embed_mean=(hidden_states*text_masks.unsqueeze(-1)).sum(1)/text_masks.sum(1).unsqueeze(-1)
        # print(embed_mean.shape)
        # ocr = torch.randn(len(inputs['ocr_input']),32,768).to(device)
        # for i in range(len(inputs['ocr_input'])):
        #     ocr[i] = self.bert(inputs['ocr_input'][i], inputs['ocr_mask'][i])['pooler_output']
        # fusion = torch.cat((inputs['frame_input'],ocr),dim=2)
        # fusion=self.fusion_dropout(fusion)
        # fusion_output = self.fc1(fusion)
        # fusion_embedding = self.nextvlad(fusion_output, inputs['frame_mask'])
        # fusion_embedding = self.enhance(fusion_embedding)
        # vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
        # vision_embedding = self.enhance(vision_embedding)
        # final_embedding = self.fusion([vision_embedding, bert_embedding])
        #vision_out = self.fc(vision_embedding)
        #final_embedding = torch.matmul(final_embedding,vision_out.transpose(0,1))    ##1111     
        prediction = self.fc(encoder_out).to(device)

        if inference:
            if self.is5zhe:
                return prediction
            else:
                return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'].to(device))

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softattn = SoftAttention(channels)
    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x1 = torch.mul(x, gates)
        x2 = self.softattn(x)
        
        return x1,x2

class SoftAttention(nn.Module):
    def __init__(self,hidden_size):
        super(SoftAttention,self).__init__()
        self.attn = nn.Linear(hidden_size,1)
        
    def get_attn(self,reps,mask=None):
        reps = torch.unsqueeze(reps,1)
        attn_scores = self.attn(reps).squeeze(2)
        if mask is not None:
            attn_scores = mask*attn_scores
        attn_weights = attn_scores.unsqueeze(2)
        attn_out = torch.sum(reps*attn_weights,dim=1)
        
        return attn_out
        

    def forward(self,reps,mask=None):
        attn_out = self.get_attn(reps,mask)
        
        return attn_out

class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding

    

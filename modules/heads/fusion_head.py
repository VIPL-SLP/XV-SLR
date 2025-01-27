import torch
import torch.nn as nn
from utils.positional_encoding import PositionalEncoding
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from modules.utils.slr_utils import NormBothLinear

class FusionHead(nn.Module):
    def __init__(self, in_dim, class_num, feat_type, norm_classifier=True,norm_scale=32,ratio=None):
        super(FusionHead, self).__init__()
        self.part_num = 3
        self.s_ratio = ratio['s_ratio']
        self.r_ratio = ratio['r_ratio']
        self.d_ratio = ratio['d_ratio']

        hidden_dim = 1024
        # define the linear mapping of input features
        self.fc_rgb = nn.Linear(768, hidden_dim)
        self.fc_depth = nn.Linear(768, hidden_dim)
        
        # fixed positional encoding
        self.pe = PositionalEncoding(hidden_dim)

        self.skeleton_modality_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.rgb_modality_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.depth_modality_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        transformer_layer = TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8,
            batch_first=True 
        )
        self.transformer_encoder = TransformerEncoder(
            transformer_layer, 
            num_layers=4
        )
        
        if norm_classifier:
            self.scale = norm_scale
            self.classifier = NormBothLinear(hidden_dim, class_num)
            self.classifier_s = NormBothLinear(hidden_dim, class_num)
            self.classifier_r = NormBothLinear(hidden_dim, class_num)
            self.classifier_d = NormBothLinear(hidden_dim, class_num)
        else:
            self.scale = 1
            self.classifier = nn.Linear(hidden_dim, class_num)
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def gen_mask(self,B,T,len_x):
        return torch.arange(T).expand(B, T) >= len_x.unsqueeze(1)

    def forward(self, feat_dict, len_sk, len_rgb, len_dp,device, label=None,len_x=None):
        skeleton_feat = feat_dict['2d_skeleton']
        B, T1, _ = skeleton_feat.shape
        rgb_feat = self.fc_rgb(feat_dict['rgb'])
        B, T2, _ = rgb_feat.shape # [64,9,1024]
        if 'depth' in feat_dict.keys():
            depth_feat = self.fc_depth(feat_dict['depth'])
            B, T3, _ = depth_feat.shape # [64,9,1024]

        # add modality token
        skeleton_feat = torch.cat((self.skeleton_modality_token.expand(B, 1, -1), skeleton_feat), dim=1) # [B,1+T1,hidden_dim] Passing -1 as the size for a dimension means not changing the size of that dimension.
        # add pe
        skeleton_feat = self.pe(skeleton_feat) # [B,1+T1,hidden_dim]
        # gen mask
        sk_src_key_padding_mask = self.gen_mask(B,T1,len_sk).to(device) # [B,T1][64,139]
        
        # add modality token
        rgb_feat = torch.cat((self.rgb_modality_token.expand(B, 1, -1), rgb_feat), dim=1) # [B,1+T2,hidden_dim]
        # add pe
        rgb_feat = self.pe(rgb_feat) # [64,10,1024]
        # gen mask
        rgb_src_key_padding_mask = self.gen_mask(B,T2,len_rgb).to(device) # [64,9]

        if 'depth' in feat_dict.keys():
            assert(len(depth_feat.shape) == 3)
            # add modality token
            depth_feat = torch.cat((self.depth_modality_token.expand(B, 1, -1), depth_feat), dim=1) # [B,1+T3,hidden_dim] Passing -1 as the size for a dimension means not changing the size of that dimension.
            # add pe
            depth_feat = self.pe(depth_feat) # [B,1+T1,hidden_dim]
            # gen mask
            dp_src_key_padding_mask = self.gen_mask(B,T3,len_dp).to(device) # [B,T1][64,139]

        # output cls
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        # add pe
        cls_token = self.pe(cls_token)
        
        # gen a general [B,1] shape False mask
        single_mask = torch.zeros(B, 1, dtype=torch.bool).to(device) # [B,1]

        if 'depth' in feat_dict.keys():
            # combine feat
            combined_feat = torch.cat([cls_token, skeleton_feat, rgb_feat, depth_feat], dim=1)  # [B, 3+T1+T2, hidden_dim]
            # combie mask
            src_key_padding_mask = torch.cat([single_mask,single_mask, sk_src_key_padding_mask,single_mask, rgb_src_key_padding_mask,single_mask,dp_src_key_padding_mask], dim=1)  # [B, 3+T1+T2+T3]
        
        else:
            combined_feat = torch.cat([cls_token, skeleton_feat, rgb_feat], dim=1) 
            # combie mask
            src_key_padding_mask = torch.cat([single_mask,single_mask, sk_src_key_padding_mask,single_mask, rgb_src_key_padding_mask], dim=1)  # [B, 3+T1+T2]
        
        # feature fusion
        transformer_output = self.transformer_encoder(
            combined_feat,
            src_key_padding_mask=src_key_padding_mask
            )  # [B, 1+T1+T2, 1024]
        
        cls_token_output  = transformer_output[:, 0]  # [B, 1024]

        logits = self.classifier(cls_token_output) * self.scale

        aux_logits1 = self.classifier_s(transformer_output[:, 1:1+(T1+1)].mean(dim=1)) * self.scale

        aux_logits2 = self.classifier_r(transformer_output[:, 1+(T1+1):1+(T1+1)+(T2+1)].mean(dim=1)) * self.scale

        if 'depth' in feat_dict.keys():
            aux_logits3 = self.classifier_d(transformer_output[:, 1+(T1+1)+(T2+1):].mean(dim=1)) * self.scale
        
        if self.training:

            loss_cls = self.loss_func(logits, label)
            loss_s = self.loss_func(aux_logits1, label)
            loss_r = self.loss_func(aux_logits2, label)
            if 'depth' in feat_dict.keys():
                loss_d = self.loss_func(aux_logits3, label)

                loss = loss_cls + self.s_ratio * loss_s + self.r_ratio * loss_r + self.d_ratio * loss_d
                return {
                    'loss': loss,
                    'loss_details': {
                        'celoss': loss_cls,
                        'rgb_loss': loss_r,
                        'skeleton_loss': loss_s,
                        'depth_loss': loss_d,
                        }
                }
            else:
                loss = loss_cls + self.s_ratio * loss_s + self.r_ratio * loss_r
                return {
                    'loss': loss,
                    'loss_details': {
                        'celoss': loss_cls,
                        'rgb_loss': loss_r,
                        'skeleton_loss': loss_s,
                        }
                }
        else:
            if 'depth' in feat_dict.keys():
                return {'logits': (logits+aux_logits1+aux_logits2+aux_logits3)/4}
            else:
                return {'logits': (logits+aux_logits1+aux_logits2)/3}

class ClassifyHead(nn.Module):
    def __init__(self, in_dim, class_num, feat_type, norm_classifier=True,norm_scale=32) -> None:
        super().__init__()
        self.feat_type = feat_type
        if norm_classifier:
            self.scale = norm_scale
            self.classifier = NormBothLinear(in_dim, class_num)
        else:
            self.scale = 1
            self.classifier = nn.Linear(in_dim, class_num)
        self.class_num = class_num
        self.loss_func = nn.CrossEntropyLoss()
        self.framewiseloss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, feat_dict, label=None, len_x=None,temporal_arg=None):
        temporaltype = temporal_arg['type']
        
        feat = feat_dict[self.feat_type] # [1,64,1024][B,T,C]
        
        B,T,C = feat.shape
        pre_feats = feat.reshape(B*T,C)
        if len_x == None:
            feat = feat.mean(1) # [1,1024][B,C]
        else: # LSTM
            feat = torch.stack([
                feat[i,:len_x[i]].sum(dim=0) / len_x[i] for i in range(B)
            ])
        logits = self.classifier(feat) * self.scale

        if self.training:
            loss = self.loss_func(logits, label)
            per_frame_logits = self.classifier(pre_feats).view(B,T,-1).permute(0,2,1) # [B,class_num,T]
            label_onehot = torch.zeros((B,self.class_num),device=label.device)
            label_onehot.scatter_(1,label.unsqueeze(1),1) # [B,class_num]
            loc_loss = 0
            if temporaltype == 'LSTM':
                framewise_labels = label_onehot.unsqueeze(2).expand(-1, -1, T)
                for i in range(B):
                    # valid frames
                    valid_len = len_x[i]
                    
                    logits_i = per_frame_logits[i, :valid_len, :]  # [valid_len, class_num]
                    labels_i = framewise_labels[i, :valid_len, :]  # [valid_len, class_num]
                    # frame-wise loss
                    loc_loss += self.framewiseloss(logits_i, labels_i).mean()

                loc_loss /= B # avg loss

            else:
                loc_loss = self.framewiseloss(
                per_frame_logits, 
                label_onehot[:,:,None].expand_as(per_frame_logits)
                ).mean()

            ttl_loss = loss * 0.8 + loc_loss * 0.2
            return {
                'loss': ttl_loss,
                'loss_details': {'celoss': loss,'loc_loss':loc_loss}
            }
        else:
            return {
                'logits': logits,
                'label': label
            }

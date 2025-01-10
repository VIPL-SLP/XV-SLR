import pdb
import torch
import numpy as np
import copy
from json import decoder
import torch, pickle, json
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict
from transformers import MBartTokenizer

def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training
    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.
    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    ones = torch.ones(size, size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0)

def _compute_dec_mask(self, tgt_pad_mask, future):
    tgt_len = tgt_pad_mask.size(-1)
    if not future:  # apply future_mask, result mask in (B, T, T)
        future_mask = torch.ones(
            [tgt_len, tgt_len],
            device=tgt_pad_mask.device,
            dtype=torch.uint8,
        )
        future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
        # BoolTensor was introduced in pytorch 1.2
        try:
            future_mask = future_mask.bool()
        except AttributeError:
            pass
        dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
    else:  # only mask padding, result mask in (B, 1, T)
        dec_mask = tgt_pad_mask
    return dec_mask

class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T=1):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t, labels):
        mask = (labels != 1).view(-1)
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='none') * self.T * self.T
        # loss = (loss.sum(dim=-1) * mask.detach()).sum()
        return loss

class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        # self.freq_weight = np.load('sign_freq.npy', allow_pickle=True).item()
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".
        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            labels = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: distributions with batch*seq_len x vocab_size
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == labels.shape
            )
        else:
            # targets: indices with batch*seq_len
            labels = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), labels
        )
        
        # self.freq_weight.keys().mean()
        # median_freq = np.median([*self.freq_weight.values()])
        # loss_weight = torch.ones_like(targets).float()
        # B, T = targets.shape
        # for i in range(B):
        #     for j in range(T):
        #         if targets[i, j].item() in self.freq_weight.keys():
        #             loss_weight[i, j] = median_freq / max(self.freq_weight[targets[i, j].item()], 1)
        # loss_weight = loss_weight ** 0
        # loss = (loss.sum(dim=-1) * loss_weight.view(-1)).sum()
        return loss
    
class XentLocalLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, gamma: float = 0.9, smoothing: float = 0.0):
        super(XentLocalLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        self.gamma = gamma
        self.max_decay = 0.5
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="none")
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction="none")

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".
        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def exp_gamma(self, epoch_idx):
        return self.max_decay * self.gamma ** epoch_idx
    
    def cosine_gamma(self, epoch_idx):
        return 0.5 * self.max_decay * (1 + np.cos(epoch_idx/80 * np.pi))

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets_idx, epoch_idx):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets_idx.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: distributions with batch*seq_len x vocab_size
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == targets.shape
            )
        else:
            # targets: indices with batch*seq_len
            targets = targets_idx.contiguous().view(-1)
        
        B, T, C = log_probs.shape
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        ).sum(-1)

        p = log_probs.exp()
        gamma = self.exp_gamma(epoch_idx)
        # gamma = self.cosine_gamma(epoch_idx)
        # gather_p = torch.pow((1 - torch.gather(p, -1, targets_idx.unsqueeze(-1))).squeeze(-1), gamma)
        gather_p = torch.pow(torch.gather(p, -1, targets_idx.unsqueeze(-1)).squeeze(-1), gamma)
        # prefix_p = torch.ones_like(gather_p)
        # for i in range(1, T):
        #     prefix_p[:, i] = gather_p[:, i-1] * prefix_p[:, i-1]
        loss = loss.view(B, T)
        # loss_backup = loss.clone()
        loss[:, 1:] = loss[:, 1:] * gather_p[:, :-1].detach()
        # pdb.set_trace()
        # for i in range(1, T):
        #     loss[:, i:] = loss[:, i:] * gather_p[:, i-1].unsqueeze(-1).detach()
        return loss.sum()
    


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, ignore_index: int=-100):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    for ii,ind in enumerate(index_of_eos.squeeze(-1)):
        input_ids[ii, ind:] = ignore_index
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

class BaseTokenizer(object):
    def __init__(self, tokenizer_cfg):
        self.tokenizer_cfg = tokenizer_cfg
    def __call__(self, input_str):
        pass


class TextTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)

        self.level = tokenizer_cfg.get('level','sentencepiece')
        if self.level == 'word':
            self.min_freq = tokenizer_cfg.get('min_freq',0)
            with open(tokenizer_cfg['tokenizer_file'],'r') as f:
                tokenizer_info = json.load(f)
            self.word2fre, self.special_tokens = tokenizer_info['word2fre'], tokenizer_info['special_tokens']
            self.id2token = self.special_tokens[:]
            for w in sorted(self.word2fre.keys(), key=lambda w: self.word2fre[w])[::-1]:
                f = self.word2fre[w]
                if f>=self.min_freq:
                    self.id2token.append(w)
            self.token2id = {t: id_ for id_, t in enumerate(self.id2token)}
            self.pad_index, self.eos_index, self.unk_index, self.sos_index = \
                self.token2id['<pad>'], self.token2id['</s>'], self.token2id['<unk>'], self.token2id['<s>']
            self.token2id = defaultdict(lambda:self.unk_index, self.token2id)
            self.ignore_index = self.pad_index
        elif self.level == 'sentencepiece':
            self.tokenizer = MBartTokenizer.from_pretrained(
                **tokenizer_cfg) #tgt_lang
            self.pad_index = self.tokenizer.convert_tokens_to_ids('<pad>')
            self.ignore_index = self.pad_index
            
            self.pruneids_file = tokenizer_cfg['pruneids_file']
            with open(self.pruneids_file, 'rb') as f:
                self.pruneids = pickle.load(f) # map old2new #gls2token
                for t in ['<pad>','<s>','</s>','<unk>']:
                    id_ = self.tokenizer.convert_tokens_to_ids(t)
                    assert self.pruneids[id_] == id_, '{}->{}'.format(id_, self.pruneids[id_])
            self.pruneids_reverse = {i2:i1 for i1,i2 in self.pruneids.items()}
            self.lang_index = self.pruneids[self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)]
            self.sos_index = self.lang_index
            self.eos_index = self.pruneids[self.tokenizer.convert_tokens_to_ids('</s>')]
        else:
            raise ValueError

    def generate_decoder_labels(self, input_ids):
        decoder_labels = torch.where(
            input_ids==self.lang_index,  #already be mapped into pruned_vocab
            torch.ones_like(input_ids)*self.ignore_index, input_ids)
        return decoder_labels

    def generate_decoder_inputs(self, input_ids):
        decoder_inputs = shift_tokens_right(input_ids, 
            pad_token_id=self.pad_index,
            ignore_index=self.pad_index)
        return decoder_inputs

    def prune(self, input_ids):
        pruned_input_ids = []
        for  single_seq in input_ids:
            pruned_single_seq = []
            for id_ in single_seq:
                if not id_ in self.pruneids:
                    new_id = self.pruneids[self.tokenizer.convert_tokens_to_ids('<unk>')]
                    print(id_)
                    print(self.tokenizer.convert_ids_to_tokens(id_))
                else:
                    new_id = self.pruneids[id_]
                pruned_single_seq.append(new_id)
            pruned_input_ids.append(pruned_single_seq)
        return torch.tensor(pruned_input_ids, dtype=torch.long)
    
    def prune_reverse(self, pruned_input_ids):
        batch_size, max_len = pruned_input_ids.shape
        input_ids = pruned_input_ids.clone()
        for b in range(batch_size):
            for i in range(max_len):
                id_ = input_ids[b,i].item()
                if not id_ in self.pruneids_reverse:
                    new_id = self.tokenizer.convert_tokens_to_ids('<unk>')
                else:
                    new_id = self.pruneids_reverse[id_]
                input_ids[b,i] = new_id
        return input_ids
    
    def __call__(self, input_str):
        if self.level == 'sentencepiece':
            with self.tokenizer.as_target_tokenizer():
                raw_outputs = self.tokenizer(input_str, 
                    #return_tensors="pt", 
                    return_attention_mask=True,
                    return_length=True,
                    padding='longest')
            outputs = {}
            pruned_input_ids = self.prune(raw_outputs['input_ids'])
            outputs['labels'] = self.generate_decoder_labels(pruned_input_ids)
            outputs['decoder_input_ids'] = self.generate_decoder_inputs(pruned_input_ids)
        elif self.level == 'word':
            #input as a batch
            batch_labels, batch_decoder_input_ids, batch_lengths = [],[],[]
            for text in input_str:
                labels, decoder_input_ids = [], [self.sos_index]
                for ti, t in enumerate(text.split()):
                    id_ = self.token2id[t]
                    labels.append(id_)
                    decoder_input_ids.append(id_)
                labels.append(self.eos_index)
                batch_labels.append(labels)
                batch_decoder_input_ids.append(decoder_input_ids)
                batch_lengths.append(len(labels))
            #padding
            max_length = max(batch_lengths)
            padded_batch_labels, padded_batch_decoder_input_ids = [], []
            for labels, decoder_input_ids in zip(batch_labels, batch_decoder_input_ids):
                padded_labels = labels + [self.pad_index]*(max_length-len(labels))
                padded_decoder_input_ids = decoder_input_ids + [self.ignore_index]*(max_length-len(decoder_input_ids))
                assert len(padded_labels)==len(padded_decoder_input_ids)
                padded_batch_labels.append(padded_labels)
                padded_batch_decoder_input_ids.append(padded_decoder_input_ids)
            outputs = {
                'labels': torch.tensor(padded_batch_labels, dtype=torch.long),
                'decoder_input_ids': torch.tensor(padded_batch_decoder_input_ids, dtype=torch.long)
            }
        else:
            raise ValueError
        return outputs 
    
    def batch_decode(self, sequences):
        #remove the first token (bos)
        sequences = sequences[:,1:]
        if self.level == 'sentencepiece':
            sequences_ = self.prune_reverse(sequences)
            decoded_sequences = self.tokenizer.batch_decode(sequences_, skip_special_tokens=True)
            if 'de' in self.tokenizer.tgt_lang:
                for di, d in enumerate(decoded_sequences):
                    if len(d)>2 and d[-1]=='.' and d[-2]!=' ':
                        d = d[:-1]+ ' .'
                        decoded_sequences[di] = d 
        elif self.level == 'word':
            #... .</s>
            decoded_sequences = [' '.join([self.id2token[s] for s in seq]) for seq in sequences]
        else:
            raise ValueError
        return decoded_sequences

class SimpleTextTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)

        self.tokenizer = MBartTokenizer.from_pretrained(
            **tokenizer_cfg) #tgt_lang
        self.pad_index = self.tokenizer.convert_tokens_to_ids('<pad>')
        self.ignore_index = self.pad_index

        self.lang_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang)
        self.sos_index = self.lang_index
        self.eos_index = self.tokenizer.convert_tokens_to_ids('</s>')
    
    def generate_decoder_labels(self, input_ids):
        # mask lang tag directly which will not be considered in the loss function
        decoder_labels = torch.where(
            input_ids==self.lang_index,  #already be mapped into pruned_vocab
            torch.ones_like(input_ids)*self.ignore_index, input_ids)
        return decoder_labels

    def generate_decoder_inputs(self, input_ids):
        decoder_inputs = shift_tokens_right(input_ids, 
            pad_token_id=self.pad_index,
            ignore_index=self.pad_index)
        return decoder_inputs
    
    def __call__(self, input_str):
        with self.tokenizer.as_target_tokenizer():
            raw_outputs = self.tokenizer(input_str, 
                return_tensors="pt",
                return_attention_mask=True,
                return_length=True,
                padding='longest')
            # notice the return is tokens, <s/>, <lang_tg>
            outputs = {}
            outputs['input_ids'] = copy.deepcopy(raw_outputs['input_ids'])
            outputs['attention_mask'] = raw_outputs['attention_mask']
            outputs['labels'] = self.generate_decoder_labels(raw_outputs['input_ids'])
            outputs['decoder_input_ids'] = self.generate_decoder_inputs(raw_outputs['input_ids'])
            # notice that when generate decoder inputs, shift token operation will change the original input_ids
            # so if want to keep input_ids, you should use deepcopy and before generate decoder inputs like what is done in line 441
        return outputs

    def batch_decode(self, sequences):
        #remove the first token (bos)
        # sequences = sequences[:,1:]
        decoded_sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        if 'de' in self.tokenizer.tgt_lang:
            for di, d in enumerate(decoded_sequences):
                if len(d)>2 and d[-1]=='.' and d[-2]!=' ':
                    d = d[:-1]+ ' .'
                    decoded_sequences[di] = d
        return decoded_sequences

class BaseGlossTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        with open(tokenizer_cfg['gloss2id_file'],'rb') as f:
            self.gloss2id = pickle.load(f) #
        self.gloss2id = defaultdict(lambda: self.gloss2id['<unk>'], self.gloss2id)
        self.id2gloss = {}
        for gls, id_ in self.gloss2id.items():
            self.id2gloss[id_] = gls        
        self.lower_case = tokenizer_cfg.get('lower_case',True)
        
    def convert_tokens_to_ids(self, tokens):
        if type(tokens)==list:
            return [self.convert_tokens_to_ids(t) for t in tokens]
        else:
            return self.gloss2id[tokens]

    def convert_ids_to_tokens(self, ids):
        if type(ids)==list:
            return [self.convert_ids_to_tokens(i) for i in ids]
        else:
            return self.id2gloss[ids]
    
    def __len__(self):
        return len(self.id2gloss)


class GlossTokenizer_S2G(BaseGlossTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        if '<s>' in self.gloss2id:
            self.silence_token = '<s>'
            self.silence_id = self.convert_tokens_to_ids(self.silence_token)
        elif '<si>' in self.gloss2id:
            self.silence_token = '<si>'
            self.silence_id = self.convert_tokens_to_ids(self.silence_token)
        else:
            raise ValueError            
        assert self.silence_id==0, (self.silence_id)
        self.pad_token = '<pad>'
        self.pad_id = self.convert_tokens_to_ids(self.pad_token)

    def __call__(self, batch_gls_seq):
        max_length = max([len(gls_seq.split()) for gls_seq in batch_gls_seq])
        gls_lengths, batch_gls_ids = [], []
        for ii, gls_seq in enumerate(batch_gls_seq):
            gls_ids = [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in gls_seq.split()]
            gls_lengths.append(len(gls_ids))
            gls_ids = gls_ids+(max_length-len(gls_ids))*[self.pad_id]
            batch_gls_ids.append(gls_ids)
        gls_lengths = torch.tensor(gls_lengths)
        batch_gls_ids = torch.tensor(batch_gls_ids)
        return {'gls_lengths':gls_lengths, 'gloss_labels': batch_gls_ids}
        

class GlossTokenizer_G2T(BaseGlossTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        self.src_lang = tokenizer_cfg['src_lang']
    def __call__(self, batch_gls_seq):
        #batch
        max_length = max([len(gls_seq.split()) for gls_seq in batch_gls_seq])+2 #include </s> <lang>
        batch_gls_ids = []
        attention_mask = torch.zeros([len(batch_gls_seq), max_length], dtype=torch.long)
        for ii, gls_seq in enumerate(batch_gls_seq):
            gls_ids = [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in gls_seq.split()]
            #add </s> <lang> + padding
            gls_ids = gls_ids + [self.gloss2id['</s>'],self.gloss2id[self.src_lang]]
            attention_mask[ii,:len(gls_ids)] = 1
            gls_ids = gls_ids + (max_length-len(gls_ids))*[self.gloss2id['<pad>']]
            batch_gls_ids.append(gls_ids)
        input_ids = torch.tensor(batch_gls_ids, dtype=torch.long)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {'input_ids':input_ids, 'attention_mask':attention_mask}

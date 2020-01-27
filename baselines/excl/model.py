import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import RNNEncoder
from easydict import EasyDict as edict


excl_base_cfg = edict(
    visual_input_size=2048,  # changes based on visual input type
    query_input_size=768,
    sub_input_size=768,
    hidden_size=256,  #
    drop=0.5,  # dropout for other layers
    ctx_mode="video_sub",  # which context are used. 'video', 'sub' or 'video_sub'
    initializer_range=0.02,
)


class EXCL(nn.Module):
    def __init__(self, config):
        super(EXCL, self).__init__()
        self.config = config
        self.use_video = "video" in config.ctx_mode
        self.use_sub = "sub" in config.ctx_mode

        self.query_encoder = RNNEncoder(
            word_embedding_size=config.query_input_size,
            hidden_size=config.hidden_size // 2,
            bidirectional=True,
            n_layers=1,
            rnn_type="lstm",
            return_outputs=False,
            return_hidden=True
        )

        if self.use_video:
            self.video_encoder = RNNEncoder(
                word_embedding_size=config.visual_input_size,
                hidden_size=config.hidden_size // 2,
                bidirectional=True,
                n_layers=1,
                rnn_type="lstm",
                return_outputs=True,
                return_hidden=False)

            self.video_encoder2 = RNNEncoder(
                word_embedding_size=2*config.hidden_size,
                hidden_size=config.hidden_size // 2,
                bidirectional=True,
                n_layers=1,
                rnn_type="lstm",
                return_outputs=True,
                return_hidden=False)

            self.video_st_predictor = nn.Sequential(
                nn.Linear(3*config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1))
            self.video_ed_predictor = copy.deepcopy(self.video_st_predictor)

        if self.use_sub:
            self.sub_encoder = RNNEncoder(
                word_embedding_size=config.sub_input_size,
                hidden_size=config.hidden_size // 2,
                bidirectional=True,
                n_layers=1,
                rnn_type="lstm",
                return_outputs=True,
                return_hidden=False)

            self.sub_encoder2 = RNNEncoder(
                word_embedding_size=2*config.hidden_size,
                hidden_size=config.hidden_size // 2,
                bidirectional=True,
                n_layers=1,
                rnn_type="lstm",
                return_outputs=True,
                return_hidden=False)

            self.sub_st_predictor = nn.Sequential(
                nn.Linear(3*config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1))
            self.sub_ed_predictor = copy.deepcopy(self.video_st_predictor)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def get_prob_single_stream(self, encoded_query, ctx_feat, ctx_mask, module_name=None):
        ctx_mask_rnn = ctx_mask.sum(1).long()
        ctx_feat1 = getattr(self, module_name+"_encoder")(
            F.dropout(ctx_feat, p=self.config.drop, training=self.training),
            ctx_mask_rnn)[0]  # (N, Lc, D)
        ctx_feat2 = getattr(self, module_name+"_encoder2")(
            F.dropout(torch.cat([ctx_feat1, encoded_query], dim=-1), p=self.config.drop, training=self.training),
            ctx_mask_rnn)[0]  # (N, Lc, D)
        ctx_feat3 = torch.cat([ctx_feat2, ctx_feat1, encoded_query], dim=2)  # (N, Lc, 3D)
        st_probs = getattr(self, module_name+"_st_predictor")(ctx_feat3).squeeze()  # (N, Lc)
        ed_probs = getattr(self, module_name+"_ed_predictor")(ctx_feat3).squeeze()  # (N, Lc)
        st_probs = mask_logits(st_probs, ctx_mask)
        ed_probs = mask_logits(ed_probs, ctx_mask)
        return st_probs, ed_probs

    def forward(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask,
                tef_feat, tef_mask, st_ed_indices, is_training=True):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            tef_feat: (N, Lv, 2) or None,
            tef_mask: (N, Lv) or None,
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed labels respectively.
            is_training:
        """
        query_mask = query_mask.sum(1).long()
        encoded_query = self.query_encoder(query_feat, query_mask)[1]   # (N, D)
        encoded_query = encoded_query.unsqueeze(1).repeat(1, video_feat.shape[1], 1)  # (N, Lc, D)

        video_st_prob, video_ed_prob = self.get_prob_single_stream(
            encoded_query, video_feat, video_mask, module_name="video") if self.use_video else (0, 0)

        sub_st_prob, sub_ed_prob = self.get_prob_single_stream(
            encoded_query, sub_feat, sub_mask, module_name="sub") if self.use_sub else (0, 0)

        st_prob = (video_st_prob + sub_st_prob) / (self.use_video + self.use_sub)
        ed_prob = (video_ed_prob + sub_ed_prob) / (self.use_video + self.use_sub)

        if is_training:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed

            return loss_st_ed, {"loss_st_ed": float(loss_st_ed)}, st_prob, ed_prob
        else:
            # used to measure the runtime. not useful for other experiments.
            prob_product = torch.einsum("bm,bn->bmn", st_prob, ed_prob)  # (N, L, L)
            prob_product = torch.triu(prob_product)  # ()
            prob_product = prob_product.view(prob_product.shape[0], -1)
            prob_product = torch.topk(prob_product, k=100, dim=1, largest=True)
            return None


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)

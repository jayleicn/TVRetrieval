import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import RNNEncoder
from easydict import EasyDict as edict


cal_base_cfg = edict(
    visual_input_size=2048,  # changes based on visual input type
    textual_input_size=768,
    query_feat_size=768,
    visual_hidden_size=500,  #
    output_size=100,
    embedding_size=768,
    lstm_hidden_size=1000,
    margin=0.1,  # margin for ranking loss
    loss_type="hinge",  # loss type, 'hinge' or 'lse'
    inter_loss_weight=0.4,  # weight for inter negatives
    ctx_mode="video"
)


class CAL(nn.Module):
    def __init__(self, config):
        super(CAL, self).__init__()
        self.config = config

        self.moment_mlp = nn.Sequential(
            nn.Linear(config.visual_input_size, config.visual_hidden_size),
            nn.ReLU(True),
            nn.Linear(config.visual_hidden_size, config.output_size),
        )

        self.query_lstm = RNNEncoder(word_embedding_size=config.embedding_size,
                                     hidden_size=config.lstm_hidden_size,
                                     bidirectional=False,
                                     rnn_type="lstm",
                                     dropout_p=0,
                                     n_layers=1,
                                     return_outputs=False)

        self.query_linear = nn.Linear(config.lstm_hidden_size, config.output_size)

    def moment_encoder(self, moment_feat):
        """moment_feat: (N, L_clip, D_v)"""
        return F.normalize(self.moment_mlp(moment_feat), p=2, dim=-1)  # (N, L_clip, D_o)

    def query_encoder(self, query_feat, query_mask):
        """
        Args:
            query_feat: (N, L_q, D_q), torch.float32
            query_mask: (N, L_q), torch.float32, with 1 indicates valid query, 0 indicates mask
        """
        _, hidden = self.query_lstm(query_feat, torch.sum(query_mask, dim=1).long())
        return F.normalize(self.query_linear(hidden), p=2, dim=-1)  # (N, D_o)

    def compute_pdist(self, query_embedding, moment_feat, moment_mask):
        """ pairwise L2 distance
        Args:
            query_embedding: (N, D_o)
            moment_feat: (N, L_clip, D_v)
            moment_mask: (N, L_clip), torch.float32, where 1 indicates valid, 0 indicates padding
        """
        moment_embedding = self.moment_encoder(moment_feat)  # (N, L_clip, D_o)
        moment_clip_dist = torch.sum((moment_embedding - query_embedding.unsqueeze(1)) ** 2, dim=2)  # (N, L_clip)
        moment_dist = torch.sum(moment_clip_dist * moment_mask, dim=1) / moment_mask.sum(1)  # (N, )
        return moment_dist  # (N, )

    @classmethod
    def compute_cdist_inference(cls, query_embeddings, moment_embeddings, moment_mask):
        """ Compute L2 distance for every possible pair of queries and proposals. This is different from
        compute_pdist as the latter computes only pairs at each row.
        Args:
            query_embeddings: (N_q, D_o)
            moment_embeddings: (N_prop, N_clips, D_o)
            moment_mask: (N_prop, N_clips)
        return:
            query_moment_scores: (N_q, N_prop)
        """
        # sync device
        query_device = query_embeddings.device  # convert to cuda if we want to use GPU
        if moment_embeddings.device != query_device:
            moment_embeddings = moment_embeddings.to(query_device)
            moment_mask = moment_mask.to(query_device)

        # compute
        n_query = query_embeddings.shape[0]
        n_prop, n_clips, d = moment_embeddings.shape
        query_clip_dist = torch.cdist(
            query_embeddings, moment_embeddings.reshape(-1, d), p=2) ** 2  # (N_q, N_prop * N_clips)
        query_clip_dist = query_clip_dist.reshape(n_query, n_prop, n_clips)
        query_moment_dist = torch.sum(
            query_clip_dist * moment_mask.unsqueeze(0), dim=2) / moment_mask.sum(1).unsqueeze(0)
        return query_moment_dist  # (N_q, N_prop)

    def forward(self, query_feat, query_mask, pos_moment_feat, pos_moment_mask,
                intra_neg_moment_feat, intra_neg_moment_mask,
                inter_neg_moment_feat, inter_neg_moment_mask):
        """
        Args:
            query_feat: (N, L, D_q)
            query_mask: (N, L)
            pos_moment_feat: (N, L_clip_1, D_v)
            pos_moment_mask: (N, L_clip_1)
            intra_neg_moment_feat: (N, L_clip_2, D_v)
            intra_neg_moment_mask: (N, L_clip_2)
            inter_neg_moment_feat: (N, L_clip_3, D_v)
            inter_neg_moment_mask: (N, L_clip_2)
        """
        query_embed = self.query_encoder(query_feat, query_mask)  # (N, D_o)
        pos_dist = self.compute_pdist(query_embed, pos_moment_feat, pos_moment_mask)  # (N, )
        intra_neg_dist = self.compute_pdist(query_embed, intra_neg_moment_feat, intra_neg_moment_mask)  # (N, )
        if self.config.inter_loss_weight == 0:  # should be zero for tef_only method.
            loss_inter = 0.
        else:
            inter_neg_dist = self.compute_pdist(query_embed, inter_neg_moment_feat, inter_neg_moment_mask)  # (N, )
            loss_inter = self.calc_loss(pos_dist, inter_neg_dist)

        loss = self.calc_loss(pos_dist, intra_neg_dist) + self.config.inter_loss_weight * loss_inter
        return loss

    def calc_loss(self, pos_dist, neg_dist):
        """ Note here we encourage positive distance to be smaller than negative distance.
        Args:
            pos_dist: (N, ), torch.float32
            neg_dist: (N, ), torch.float32
        """
        if self.config.loss_type == "hinge":  # max(0, m + S_pos - S_neg)
            return torch.clamp(self.config.margin + pos_dist - neg_dist, min=0).sum() / len(pos_dist)
        elif self.config.loss_type == "lse":  # log[1 + exp(S_pos - S_neg)]
            return torch.log1p(torch.exp(pos_dist - neg_dist)).sum() / len(pos_dist)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


class CALWithSub(nn.Module):
    def __init__(self, config):
        super(CALWithSub, self).__init__()
        self.config = config
        self.use_video = "video" in config.ctx_mode
        self.use_sub = "sub" in config.ctx_mode
        self.use_tef = "tef" in config.ctx_mode
        self.tef_only = self.use_tef and not self.use_video and not self.use_sub

        if self.use_video or self.tef_only:
            self.video_moment_mlp = nn.Sequential(
                nn.Linear(config.visual_input_size, config.visual_hidden_size),
                nn.ReLU(True),
                nn.Linear(config.visual_hidden_size, config.output_size),
            )

        if self.use_sub:
            self.sub_moment_mlp = nn.Sequential(
                nn.Linear(config.textual_input_size, config.visual_hidden_size),
                nn.ReLU(True),
                nn.Linear(config.visual_hidden_size, config.output_size),
            )

        self.query_lstm = RNNEncoder(word_embedding_size=config.query_feat_size,
                                     hidden_size=config.lstm_hidden_size,
                                     bidirectional=False,
                                     rnn_type="lstm",
                                     dropout_p=0,
                                     n_layers=1,
                                     return_outputs=False)

        self.query_linear = nn.Linear(config.lstm_hidden_size, config.output_size)

    def moment_encoder(self, moment_feat, module_name="video"):
        """moment_feat: (N, L_clip, D_v)"""
        if moment_feat is not None:
            encoder = getattr(self, module_name + "_moment_mlp")
            return F.normalize(encoder(moment_feat), p=2, dim=-1)  # (N, L_clip, D_o)
        else:
            return None

    def query_encoder(self, query_feat, query_mask):
        """
        Args:
            query_feat: (N, L_q, D_q), torch.float32
            query_mask: (N, L_q), torch.float32, with 1 indicates valid query, 0 indicates mask
        """
        _, hidden = self.query_lstm(query_feat, torch.sum(query_mask, dim=1).long())
        return F.normalize(self.query_linear(hidden), p=2, dim=-1)  # (N, D_o)

    def _compute_pdist(self, query_embedding, moment_feat, moment_mask, module_name="video"):
        """ pairwise L2 distance
        Args:
            query_embedding: (N, D_o)
            moment_feat: (N, L_clip, D_v)
            moment_mask: (N, L_clip), torch.float32, where 1 indicates valid, 0 indicates padding
        """
        moment_embedding = self.moment_encoder(moment_feat, module_name=module_name)  # (N, L_clip, D_o)
        moment_clip_dist = torch.sum((moment_embedding - query_embedding.unsqueeze(1)) ** 2, dim=2)  # (N, L_clip)
        moment_dist = torch.sum(moment_clip_dist * moment_mask, dim=1) / moment_mask.sum(1)  # (N, )
        return moment_dist  # (N, )

    def compute_pdist(self, query_embedding, moment_video_feat, moment_sub_feat, moment_mask):
        """ pairwise L2 distance
        Args:
            query_embedding: (N, D_o)
            moment_video_feat: (N, L_clip, D_v)
            moment_sub_feat: (N, L_clip, D_t)
            moment_mask: (N, L_clip), torch.float32, where 1 indicates valid, 0 indicates padding
        """
        divisor = (self.use_video or self.tef_only) + self.use_sub
        video_moment_dist = self._compute_pdist(query_embedding, moment_video_feat, moment_mask, module_name="video") \
            if self.use_video or self.tef_only else 0
        sub_moment_dist = self._compute_pdist(query_embedding, moment_sub_feat, moment_mask, module_name="sub") \
            if self.use_sub else 0
        return (video_moment_dist + sub_moment_dist) / divisor  # (N, )

    def _compute_cdist_inference(self, query_embeddings, moment_embeddings, moment_mask):
        """ Compute L2 distance for every possible pair of queries and proposals. This is different from
        compute_pdist as the latter computes only pairs at each row.
        Args:
            query_embeddings: (N_q, D_o)
            moment_embeddings: (N_prop, N_clips, D_o)
            moment_mask: (N_prop, N_clips)
        return:
            query_moment_scores: (N_q, N_prop)
        """
        # sync device
        query_device = query_embeddings.device  # convert to cuda if we want to use GPU
        if moment_embeddings.device != query_device:
            moment_embeddings = moment_embeddings.to(query_device)
            moment_mask = moment_mask.to(query_device)

        # compute
        n_query = query_embeddings.shape[0]
        n_prop, n_clips, d = moment_embeddings.shape
        query_clip_dist = torch.cdist(
            query_embeddings, moment_embeddings.reshape(-1, d), p=2) ** 2  # (N_q, N_prop * N_clips)
        query_clip_dist = query_clip_dist.reshape(n_query, n_prop, n_clips)
        query_moment_dist = torch.sum(
            query_clip_dist * moment_mask.unsqueeze(0), dim=2) / moment_mask.sum(1).unsqueeze(0)
        return query_moment_dist  # (N_q, N_prop)

    def compute_cdist_inference(self, query_embeddings, video_moment_embeddings, sub_moment_embeddings, moment_mask):
        divisor = (self.use_video or self.tef_only) + self.use_sub
        video_moment_dist = self._compute_cdist_inference(query_embeddings, video_moment_embeddings, moment_mask) \
            if self.use_video or self.tef_only else 0
        sub_moment_dist = self._compute_cdist_inference(query_embeddings, sub_moment_embeddings, moment_mask) \
            if self.use_sub else 0
        return (video_moment_dist + sub_moment_dist) / divisor  # (N_q, N_prop)

    def forward(self, query_feat, query_mask, pos_moment_video_feat, pos_moment_video_mask,
                intra_neg_moment_video_feat, intra_neg_moment_video_mask,
                inter_neg_moment_video_feat, inter_neg_moment_video_mask,
                pos_moment_sub_feat, pos_moment_sub_mask,
                intra_neg_moment_sub_feat, intra_neg_moment_sub_mask,
                inter_neg_moment_sub_feat, inter_neg_moment_sub_mask):
        """
        Args:
            query_feat: (N, L, D_q)
            query_mask: (N, L)
            pos_moment_video_feat: (N, L_clip_1, D_v)
            pos_moment_video_mask: (N, L_clip_1)
            intra_neg_moment_video_feat: (N, L_clip_2, D_v)
            intra_neg_moment_video_mask: (N, L_clip_2)
            inter_neg_moment_video_feat: (N, L_clip_3, D_v)
            inter_neg_moment_video_mask: (N, L_clip_2)
            pos_moment_sub_feat:
            pos_moment_sub_mask:
            intra_neg_moment_sub_feat:
            intra_neg_moment_sub_mask:
            inter_neg_moment_sub_feat:
            inter_neg_moment_sub_mask:
        """
        query_embed = self.query_encoder(query_feat, query_mask)  # (N, D_o)
        pos_dist = self.compute_pdist(
            query_embed, pos_moment_video_feat, pos_moment_sub_feat,
            moment_mask=pos_moment_sub_mask if self.use_sub else pos_moment_video_mask)  # (N, )
        intra_neg_dist = self.compute_pdist(
            query_embed, intra_neg_moment_video_feat, intra_neg_moment_sub_feat,
            moment_mask=intra_neg_moment_sub_mask if self.use_sub else intra_neg_moment_video_mask)  # (N, )
        if self.config.inter_loss_weight == 0:  # should be zero for tef_only method.
            loss_inter = 0.
        else:
            inter_neg_dist = self.compute_pdist(
                query_embed, inter_neg_moment_video_feat, inter_neg_moment_sub_feat,
                moment_mask=inter_neg_moment_sub_mask if self.use_sub else inter_neg_moment_video_mask)  # (N, )
            loss_inter = self.calc_loss(pos_dist, inter_neg_dist)

        loss = self.calc_loss(pos_dist, intra_neg_dist) + self.config.inter_loss_weight * loss_inter
        return loss

    def calc_loss(self, pos_dist, neg_dist):
        """ Note here we encourage positive distance to be smaller than negative distance.
        Args:
            pos_dist: (N, ), torch.float32
            neg_dist: (N, ), torch.float32
        """
        if self.config.loss_type == "hinge":  # max(0, m + S_pos - S_neg)
            return torch.clamp(self.config.margin + pos_dist - neg_dist, min=0).sum() / len(pos_dist)
        elif self.config.loss_type == "lse":  # log[1 + exp(S_pos - S_neg)]
            return torch.log1p(torch.exp(pos_dist - neg_dist)).sum() / len(pos_dist)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")

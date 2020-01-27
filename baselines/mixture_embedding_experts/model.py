import torch
import torch.nn as nn
from baselines.mixture_embedding_experts.model_components import NetVLAD, MaxMarginRankingLoss, GatedEmbeddingUnit
from easydict import EasyDict as edict

mee_base_cfg = edict(
    ctx_mode="video",
    text_input_size=768,
    vid_input_size=1024,
    output_size=256,
    margin=0.2
)


class MEE(nn.Module):
    def __init__(self, config):
        super(MEE, self).__init__()
        self.config = config
        self.use_video = "video" in config.ctx_mode
        self.use_sub = "sub" in config.ctx_mode

        self.query_pooling = NetVLAD(feature_size=config.text_input_size, cluster_size=2)

        if self.use_sub:
            self.sub_query_gu = GatedEmbeddingUnit(input_dimension=self.query_pooling.out_dim,
                                                   output_dimension=config.output_size)
            self.sub_gu = GatedEmbeddingUnit(input_dimension=config.text_input_size,
                                             output_dimension=config.output_size)

        if self.use_video:
            self.video_query_gu = GatedEmbeddingUnit(input_dimension=self.query_pooling.out_dim,
                                                     output_dimension=config.output_size)
            self.video_gu = GatedEmbeddingUnit(input_dimension=config.vid_input_size,
                                               output_dimension=config.output_size)

        if self.use_video and self.use_sub:
            self.moe_fc = nn.Linear(self.query_pooling.out_dim, 2)  # weights

        self.max_margin_loss = MaxMarginRankingLoss(margin=config.margin)

    def forward(self, query_feat, query_mask, video_feat, sub_feat):
        """
        Args:
            query_feat: (N, L, D_q)
            query_mask: (N, L)
            video_feat: (N, Dv)
            sub_feat: (N, Dt)
        """
        pooled_query = self.query_pooling(query_feat)  # (N, Dt)
        encoded_video, encoded_sub = self.encode_context(video_feat, sub_feat)
        confusion_matrix = self.get_score_from_pooled_query_with_encoded_ctx(pooled_query, encoded_video, encoded_sub)
        return self.max_margin_loss(confusion_matrix)

    def encode_context(self, video_feat, sub_feat):
        """(N, D)"""
        encoded_video = self.video_gu(video_feat) if self.use_video else None
        encoded_sub = self.sub_gu(sub_feat) if self.use_sub else None
        return encoded_video, encoded_sub

    def compute_single_stream_scores_with_encoded_ctx(self, pooled_query, encoded_ctx, module_name="video"):
        encoded_query = getattr(self, module_name+"_query_gu")(pooled_query)  # (N, D)
        return torch.einsum("md,nd->mn", encoded_query, encoded_ctx)  # (N, N)

    def get_score_from_pooled_query_with_encoded_ctx(self, pooled_query, encoded_video, encoded_sub):
        """Nq may not equal to Nc
        Args:
            pooled_query: (Nq, Dt)
            encoded_video: (Nc, Dc)
            encoded_sub: (Nc, Dc)
        """

        video_confusion_matrix = self.compute_single_stream_scores_with_encoded_ctx(
            pooled_query, encoded_video, module_name="video") if self.use_video else 0
        sub_confusion_matrix = self.compute_single_stream_scores_with_encoded_ctx(
                pooled_query, encoded_sub, module_name="sub") if self.use_sub else 0

        if self.use_video and self.use_sub:
            stream_weights = self.moe_fc(pooled_query)  # (N, 2)
            confusion_matrix = \
                stream_weights[:, 0:1] * video_confusion_matrix + stream_weights[:, 1:2] * sub_confusion_matrix
        else:
            confusion_matrix = video_confusion_matrix + sub_confusion_matrix
        return confusion_matrix  # (Nq, Nc)


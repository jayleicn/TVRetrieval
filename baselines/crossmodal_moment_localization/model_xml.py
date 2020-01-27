import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from baselines.crossmodal_moment_localization.model_components import \
    BertAttention, PositionEncoding, LinearLayer, BertSelfAttention, TrainablePositionalEncoding, ConvEncoder
from utils.model_utils import RNNEncoder

base_bert_layer_config = dict(
    hidden_size=768,
    intermediate_size=768,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_attention_heads=4,
)

xml_base_config = edict(
    merge_two_stream=True,  # merge only the scores
    cross_att=True,  # cross-attention for video and subtitles
    span_predictor_type="conv",
    encoder_type="transformer",  # cnn, transformer, lstm, gru
    add_pe_rnn=False,  # add positional encoding for RNNs, (LSTM and GRU)
    visual_input_size=2048,  # changes based on visual input type
    query_input_size=768,
    sub_input_size=768,
    hidden_size=500,  #
    conv_kernel_size=5,  # conv kernel_size for st_ed predictor
    stack_conv_predictor_conv_kernel_sizes=-1,  # Do not use
    conv_stride=1,  #
    max_ctx_l=100,
    max_desc_l=30,
    input_drop=0.1,  # dropout for input
    drop=0.1,  # dropout for other layers
    n_heads=4,  # self attention heads
    ctx_mode="video_sub",  # which context are used. 'video', 'sub' or 'video_sub'
    margin=0.1,  # margin for ranking loss
    ranking_loss_type="hinge",  # loss type, 'hinge' or 'lse'
    lw_neg_q=1,  # loss weight for neg. query and pos. context
    lw_neg_ctx=1,  # loss weight for pos. query and neg. context
    lw_st_ed=1,  # loss weight for st ed prediction
    use_hard_negative=False,  # use hard negative at video level, we may change it during training.
    hard_pool_size=20,
    use_self_attention=True,
    no_modular=False,
    pe_type="none",  # no positional encoding
    initializer_range=0.02,
)


class XML(nn.Module):
    def __init__(self, config):
        super(XML, self).__init__()
        self.config = config
        # self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
        #                                             max_len=config.max_position_embeddings,
        #                                             pe_type=config.pe_type)
        self.query_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l,
            hidden_size=config.hidden_size, dropout=config.input_drop)
        self.ctx_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            hidden_size=config.hidden_size, dropout=config.input_drop)
        self.query_input_proj = LinearLayer(config.query_input_size,
                                            config.hidden_size,
                                            layer_norm=True,
                                            dropout=config.input_drop,
                                            relu=True)
        if config.encoder_type == "transformer":  # self-att encoder
            self.query_encoder = BertAttention(edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                attention_probs_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
            ))
        elif config.encoder_type == "cnn":
            self.query_encoder = ConvEncoder(
                kernel_size=5,
                n_filters=config.hidden_size,
                dropout=config.drop
            )
        elif config.encoder_type in ["gru", "lstm"]:
            self.query_encoder = RNNEncoder(
                word_embedding_size=config.hidden_size,
                hidden_size=config.hidden_size // 2,
                bidirectional=True,
                n_layers=1,
                rnn_type=config.encoder_type,
                return_outputs=True,
                return_hidden=False
            )

        conv_cfg = dict(in_channels=1,
                        out_channels=1,
                        kernel_size=config.conv_kernel_size,
                        stride=config.conv_stride,
                        padding=config.conv_kernel_size // 2,
                        bias=False)

        cross_att_cfg = edict(
            hidden_size=config.hidden_size,
            num_attention_heads=config.n_heads,
            attention_probs_dropout_prob=config.drop
        )

        self.use_video = "video" in config.ctx_mode
        if self.use_video:
            self.video_input_proj = LinearLayer(config.visual_input_size,
                                                config.hidden_size,
                                                layer_norm=True,
                                                dropout=config.input_drop,
                                                relu=True)
            self.video_encoder1 = copy.deepcopy(self.query_encoder)
            self.video_encoder2 = copy.deepcopy(self.query_encoder)
            if self.config.cross_att:
                self.video_cross_att = BertSelfAttention(cross_att_cfg)
                self.video_cross_layernorm = nn.LayerNorm(config.hidden_size)
            else:
                if self.config.encoder_type == "transformer":
                    self.video_encoder3 = copy.deepcopy(self.query_encoder)
            self.video_query_linear = nn.Linear(config.hidden_size, config.hidden_size)
            if config.span_predictor_type == "conv":
                if not config.merge_two_stream:
                    self.video_st_predictor = nn.Conv1d(**conv_cfg)
                    self.video_ed_predictor = nn.Conv1d(**conv_cfg)
            elif config.span_predictor_type == "cat_linear":
                self.video_st_predictor = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(2)])
                self.video_ed_predictor = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(2)])

        self.use_sub = "sub" in config.ctx_mode
        if self.use_sub:
            self.sub_input_proj = LinearLayer(config.sub_input_size,
                                              config.hidden_size,
                                              layer_norm=True,
                                              dropout=config.input_drop,
                                              relu=True)
            self.sub_encoder1 = copy.deepcopy(self.query_encoder)
            self.sub_encoder2 = copy.deepcopy(self.query_encoder)
            if self.config.cross_att:
                self.sub_cross_att = BertSelfAttention(cross_att_cfg)
                self.sub_cross_layernorm = nn.LayerNorm(config.hidden_size)
            else:
                if self.config.encoder_type == "transformer":
                    self.sub_encoder3 = copy.deepcopy(self.query_encoder)
            self.sub_query_linear = nn.Linear(config.hidden_size, config.hidden_size)
            if config.span_predictor_type == "conv":
                if not config.merge_two_stream:
                    self.sub_st_predictor = nn.Conv1d(**conv_cfg)
                    self.sub_ed_predictor = nn.Conv1d(**conv_cfg)
            elif config.span_predictor_type == "cat_linear":
                self.sub_st_predictor = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(2)])
                self.sub_ed_predictor = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(2)])

        self.modular_vector_mapping = nn.Linear(in_features=config.hidden_size,
                                                out_features=self.use_sub + self.use_video,
                                                bias=False)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")

        if config.merge_two_stream and config.span_predictor_type == "conv":
            if self.config.stack_conv_predictor_conv_kernel_sizes == -1:
                self.merged_st_predictor = nn.Conv1d(**conv_cfg)
                self.merged_ed_predictor = nn.Conv1d(**conv_cfg)
            else:
                print("Will be using  multiple Conv layers for prediction.")
                self.merged_st_predictors = nn.ModuleList()
                self.merged_ed_predictors = nn.ModuleList()
                num_convs = len(self.config.stack_conv_predictor_conv_kernel_sizes)
                for k in self.config.stack_conv_predictor_conv_kernel_sizes:
                    conv_cfg = dict(in_channels=1,
                                    out_channels=1,
                                    kernel_size=k,
                                    stride=config.conv_stride,
                                    padding=k // 2,
                                    bias=False)
                    self.merged_st_predictors.append(nn.Conv1d(**conv_cfg))
                    self.merged_ed_predictors.append(nn.Conv1d(**conv_cfg))
                self.combine_st_conv = nn.Linear(num_convs, 1, bias=False)
                self.combine_ed_conv = nn.Linear(num_convs, 1, bias=False)

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

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def set_train_st_ed(self, lw_st_ed):
        """pre-train video retrieval then span prediction"""
        self.config.lw_st_ed = lw_st_ed

    def forward(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask,
                tef_feat, tef_mask, st_ed_indices):
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
        """
        video_feat1, video_feat2, sub_feat1, sub_feat2 = \
            self.encode_context(video_feat, video_mask, sub_feat, sub_mask)

        query_context_scores, st_prob, ed_prob = \
            self.get_pred_from_raw_query(query_feat, query_mask,
                                         video_feat1, video_feat2, video_mask,
                                         sub_feat1, sub_feat2, sub_mask, cross=False)

        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed

        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(query_context_scores)

        loss_st_ed = self.config.lw_st_ed * loss_st_ed
        loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
        loss_neg_q = self.config.lw_neg_q * loss_neg_q
        loss = loss_st_ed + loss_neg_ctx + loss_neg_q
        return loss, {"loss_st_ed": float(loss_st_ed),
                      "loss_neg_ctx": float(loss_neg_ctx),
                      "loss_neg_q": float(loss_neg_q),
                      "loss_overall": float(loss)}

    def get_visualization_data(self, query_feat, query_mask, video_feat, video_mask, sub_feat, sub_mask,
                               tef_feat, tef_mask, st_ed_indices):
        assert self.config.merge_two_stream and self.use_video and self.use_sub and not self.config.no_modular
        video_feat1, video_feat2, sub_feat1, sub_feat2 = \
            self.encode_context(video_feat, video_mask, sub_feat, sub_mask)
        encoded_query = self.encode_input(query_feat, query_mask,
                                          self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
        # (N, D), (N, D), (N, L, 2)
        video_query, sub_query, modular_att_scores = \
            self.get_modularized_queries(encoded_query, query_mask, return_modular_att=True)
        # (N, L), (N, L), (N, L)
        st_prob, ed_prob, similarity_scores, video_similarity, sub_similarity = self.get_merged_st_ed_prob(
            video_query, video_feat2, sub_query, sub_feat2, video_mask, cross=False, return_similaity=True)

        # clean up invalid bits
        data = dict(modular_att_scores=modular_att_scores.cpu().numpy(),  # (N, Lq, 2), row 0, 1 are video, sub.
                    st_prob=st_prob.cpu().numpy(),  # (N, L)
                    ed_prob=ed_prob.cpu().numpy(),  # (N, L)
                    similarity_scores=similarity_scores.cpu().numpy(),  # (N, L)
                    video_similarity=video_similarity.cpu().numpy(),  # (N, L)
                    sub_similarity=sub_similarity.cpu().numpy(),  # (N, L)
                    st_ed_indices=st_ed_indices.cpu().numpy())  # (N, L)
        query_lengths = query_mask.sum(1).to(torch.long).cpu().tolist()  # (N, )
        ctx_lengths = video_mask.sum(1).to(torch.long).cpu().tolist()  # (N, )
        # print("query_lengths {}".format((type(query_lengths), len(query_lengths), query_lengths[:10])))
        for k, v in data.items():
            if k == "modular_att_scores":
                # print(k, v, v.shape, type(v))
                data[k] = [e[:l] for l, e in zip(query_lengths, v)]  # list(e) where e is  (Lq_i, 2)
            else:
                data[k] = [e[:l] for l, e in zip(ctx_lengths, v)]   # list(e) where e is (Lc_i)

        # aggregate info for each example
        datalist = []
        for idx in range(len(data["modular_att_scores"])):
            datalist.append({k: v[idx] for k, v in data.items()})
        return datalist  # list(dicts) of length N

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask,
                                          self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
        video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        return video_query, sub_query

    def non_cross_encode_context(self, context_feat, context_mask, module_name="video"):
        encoder_layer3 = getattr(self, module_name + "_encoder3") \
            if self.config.encoder_type == "transformer" else None
        return self._non_cross_encode_context(context_feat, context_mask,
                                              input_proj_layer=getattr(self, module_name + "_input_proj"),
                                              encoder_layer1=getattr(self, module_name + "_encoder1"),
                                              encoder_layer2=getattr(self, module_name + "_encoder2"),
                                              encoder_layer3=encoder_layer3)

    def _non_cross_encode_context(self, context_feat, context_mask, input_proj_layer,
                                  encoder_layer1, encoder_layer2, encoder_layer3=None):
        """
        Args:
            context_feat: (N, L, D)
            context_mask: (N, L)
            input_proj_layer:
            encoder_layer1:
            encoder_layer2:
            encoder_layer3
        """
        context_feat1 = self.encode_input(
            context_feat, context_mask, input_proj_layer, encoder_layer1, self.ctx_pos_embed)  # (N, L, D)
        if self.config.encoder_type in ["transformer", "cnn"]:
            context_mask = context_mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
            context_feat2 = encoder_layer2(context_feat1, context_mask)  # (N, L, D)
            if self.config.encoder_type == "transformer":
                context_feat2 = encoder_layer3(context_feat2, context_mask)
        elif self.config.encoder_type in ["gru", "lstm"]:
            context_mask = context_mask.sum(1).long()  # (N, ), torch.LongTensor
            context_feat2 = encoder_layer2(context_feat1, context_mask)[0]  # (N, L, D)
        else:
            raise NotImplementedError
        return context_feat1, context_feat2

    def encode_context(self, video_feat, video_mask, sub_feat, sub_mask):
        if self.config.cross_att:
            assert self.use_video and self.use_sub
            return self.cross_encode_context(video_feat, video_mask, sub_feat, sub_mask)
        else:
            video_feat1, video_feat2 = (None,) * 2
            if self.use_video:
                video_feat1, video_feat2 = self.non_cross_encode_context(video_feat, video_mask, module_name="video")
            sub_feat1, sub_feat2 = (None,) * 2
            if self.use_sub:
                sub_feat1, sub_feat2 = self.non_cross_encode_context(sub_feat, sub_mask, module_name="sub")
            return video_feat1, video_feat2, sub_feat1, sub_feat2

    def cross_encode_context(self, video_feat, video_mask, sub_feat, sub_mask):
        encoded_video_feat = self.encode_input(video_feat, video_mask,
                                               self.video_input_proj, self.video_encoder1, self.ctx_pos_embed)
        encoded_sub_feat = self.encode_input(sub_feat, sub_mask,
                                             self.sub_input_proj, self.sub_encoder1, self.ctx_pos_embed)
        x_encoded_video_feat = self.cross_context_encoder(
            encoded_video_feat, video_mask, encoded_sub_feat, sub_mask,
            self.video_cross_att, self.video_cross_layernorm, self.video_encoder2)  # (N, L, D)
        x_encoded_sub_feat = self.cross_context_encoder(
            encoded_sub_feat, sub_mask, encoded_video_feat, video_mask,
            self.sub_cross_att, self.sub_cross_layernorm, self.sub_encoder2)  # (N, L, D)
        return encoded_video_feat, x_encoded_video_feat, encoded_sub_feat, x_encoded_sub_feat

    def cross_context_encoder(self, main_context_feat, main_context_mask, side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer, self_att_layer):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer:
            norm_layer:
            self_att_layer:
        """
        cross_mask = torch.einsum("bm,bn->bmn", main_context_mask, side_context_mask)  # (N, Lq, Lk)
        cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)  # (N, Lq, D)
        residual_out = norm_layer(cross_out + main_context_feat)
        if self.config.encoder_type in ["cnn", "transformer"]:
            return self_att_layer(residual_out, main_context_mask.unsqueeze(1))
        elif self.config.encoder_type in ["gru", "lstm"]:
            return self_att_layer(residual_out, main_context_mask.sum(1).long())[0]

    def encode_input(self, feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            # add_pe: bool, whether to add positional encoding
            pos_embed_layer
        """
        feat = input_proj_layer(feat)

        if self.config.encoder_type in ["cnn", "transformer"]:
            feat = pos_embed_layer(feat)
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
            return encoder_layer(feat, mask)  # (N, L, D_hidden)
        elif self.config.encoder_type in ["gru", "lstm"]:
            if self.config.add_pe_rnn:
                feat = pos_embed_layer(feat)
            mask = mask.sum(1).long()  # (N, ), torch.LongTensor
            return encoder_layer(feat, mask)[0]  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask, return_modular_att=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        if self.config.no_modular:
            modular_query = torch.max(mask_logits(encoded_query, query_mask.unsqueeze(2)), dim=1)[0]  # (N, D)
            return modular_query, modular_query  #
        else:
            modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
            modular_attention_scores = F.softmax(
                mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
            # TODO check whether it is the same
            modular_queries = torch.einsum("blm,bld->bmd",
                                           modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
            if return_modular_att:
                assert modular_queries.shape[1] == 2
                return modular_queries[:, 0], modular_queries[:, 1], modular_attention_scores
            else:
                if modular_queries.shape[1] == 2:
                    return modular_queries[:, 0], modular_queries[:, 1]  # (N, D) * 2
                else:  # 1
                    return modular_queries[:, 0], modular_queries[:, 0]  # the same

    def get_modular_weights(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
        """
        max_encoded_query, _ = torch.max(mask_logits(encoded_query, query_mask.unsqueeze(2)), dim=1)  # (N, D)
        modular_weights = self.modular_weights_calculator(max_encoded_query)  # (N, 2)
        modular_weights = F.softmax(modular_weights, dim=-1)
        return modular_weights[:, 0:1], modular_weights[:, 1:2]  # (N, 1) * 2

    def get_video_level_scores(self, modularied_query, context_feat1, context_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat1: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat1 = F.normalize(context_feat1, dim=-1)
        query_context_scores = torch.einsum("md,nld->mln", modularied_query, context_feat1)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        query_context_scores = mask_logits(query_context_scores, context_mask)  # (N, L, N)
        query_context_scores, _ = torch.max(query_context_scores,
                                            dim=1)  # (N, N) diagonal positions are positive pairs.
        return query_context_scores

    def get_merged_st_ed_prob(self, video_query, video_feat, sub_query, sub_feat, context_mask,
                              cross=False, return_similaity=False):
        """context_mask could be either video_mask or sub_mask, since they are the same"""
        assert self.use_video and self.use_sub and self.config.span_predictor_type == "conv"
        video_query = self.video_query_linear(video_query)
        sub_query = self.sub_query_linear(sub_query)
        stack_conv = self.config.stack_conv_predictor_conv_kernel_sizes != -1
        num_convs = len(self.config.stack_conv_predictor_conv_kernel_sizes) if stack_conv else None
        if cross:
            video_similarity = torch.einsum("md,nld->mnl", video_query, video_feat)
            sub_similarity = torch.einsum("md,nld->mnl", sub_query, sub_feat)
            similarity = (video_similarity + sub_similarity) / 2  # (Nq, Nv, L)  from query to all videos.
            n_q, n_c, l = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, l)
            if not stack_conv:
                st_prob = self.merged_st_predictor(similarity).view(n_q, n_c, l)  # (Nq, Nv, L)
                ed_prob = self.merged_ed_predictor(similarity).view(n_q, n_c, l)  # (Nq, Nv, L)
            else:
                st_prob_list = []
                ed_prob_list = []
                for idx in range(num_convs):
                    st_prob_list.append(self.merged_st_predictors[idx](similarity).squeeze().unsqueeze(2))
                    ed_prob_list.append(self.merged_ed_predictors[idx](similarity).squeeze().unsqueeze(2))
                # (Nq*Nv, L, 3) --> (Nq*Nv, L) -> (Nq, Nv, L)
                st_prob = self.combine_st_conv(torch.cat(st_prob_list, dim=2)).view(n_q, n_c, l)
                ed_prob = self.combine_ed_conv(torch.cat(ed_prob_list, dim=2)).view(n_q, n_c, l)
        else:
            video_similarity = torch.einsum("bd,bld->bl", video_query, video_feat)  # (N, L)
            sub_similarity = torch.einsum("bd,bld->bl", sub_query, sub_feat)  # (N, L)
            similarity = (video_similarity + sub_similarity) / 2
            if not stack_conv:
                st_prob = self.merged_st_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
                ed_prob = self.merged_ed_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
            else:
                st_prob_list = []
                ed_prob_list = []
                for idx in range(num_convs):
                    st_prob_list.append(self.merged_st_predictors[idx](similarity.unsqueeze(1)).squeeze().unsqueeze(2))
                    ed_prob_list.append(self.merged_ed_predictors[idx](similarity.unsqueeze(1)).squeeze().unsqueeze(2))
                st_prob = self.combine_st_conv(torch.cat(st_prob_list, dim=2)).squeeze()  # (N, L, 3) --> (N, L)
                ed_prob = self.combine_ed_conv(torch.cat(ed_prob_list, dim=2)).squeeze()  # (N, L, 3) --> (N, L)
        st_prob = mask_logits(st_prob, context_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, context_mask)
        if return_similaity:
            assert not cross
            return st_prob, ed_prob, similarity, video_similarity, sub_similarity
        else:
            return st_prob, ed_prob

    def get_st_ed_prob(self, modularied_query, context_feat2, context_mask,
                       module_name="video", cross=False):
        return self._get_st_ed_prob(modularied_query, context_feat2, context_mask,
                                    module_query_linear=getattr(self, module_name + "_query_linear"),
                                    st_predictor=getattr(self, module_name + "_st_predictor"),
                                    ed_predictor=getattr(self, module_name + "_ed_predictor"),
                                    cross=cross)

    def _get_st_ed_prob(self, modularied_query, context_feat2, context_mask,
                        module_query_linear, st_predictor, ed_predictor, cross=False):
        """
        Args:
            modularied_query: (N, D)
            context_feat2: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
            module_query_linear:
            st_predictor:
            ed_predictor:
            cross: at inference, calculate prob for each possible pairs of query and context.
        """
        query = module_query_linear(modularied_query)  # (N, D) no need to normalize here.
        if cross:
            if self.config.span_predictor_type == "conv":
                similarity = torch.einsum("md,nld->mnl", query, context_feat2)  # (Nq, Nv, L)  from query to all videos.
                n_q, n_c, l = similarity.shape
                similarity = similarity.view(n_q * n_c, 1, l)
                st_prob = st_predictor(similarity).view(n_q, n_c, l)  # (Nq, Nv, L)
                ed_prob = ed_predictor(similarity).view(n_q, n_c, l)  # (Nq, Nv, L)
            elif self.config.span_predictor_type == "cat_linear":
                st_prob_q = st_predictor[0](query).unsqueeze(1)  # (Nq, 1, 1)
                st_prob_ctx = st_predictor[1](context_feat2).squeeze().unsqueeze(0)  # (1, Nv, L)
                st_prob = st_prob_q + st_prob_ctx  # (Nq, Nv, L)
                ed_prob_q = ed_predictor[0](query).unsqueeze(1)  # (Nq, 1, 1)
                ed_prob_ctx = ed_predictor[1](context_feat2).squeeze().unsqueeze(0)  # (1, Nv, L)
                ed_prob = ed_prob_q + ed_prob_ctx  # (Nq, Nv, L)
            context_mask = context_mask.unsqueeze(0)  # (1, Nv, L)
        else:
            if self.config.span_predictor_type == "conv":
                similarity = torch.einsum("bd,bld->bl", query, context_feat2)  # (N, L)
                st_prob = st_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
                ed_prob = ed_predictor(similarity.unsqueeze(1)).squeeze()  # (N, L)
            elif self.config.span_predictor_type == "cat_linear":
                # avoid concatenation by break into smaller matrix multiplications.
                st_prob = st_predictor[0](query) + st_predictor[1](context_feat2).squeeze()  # (N, L)
                ed_prob = ed_predictor[0](query) + ed_predictor[1](context_feat2).squeeze()  # (N, L)
        st_prob = mask_logits(st_prob, context_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, context_mask)
        return st_prob, ed_prob

    def get_pred_from_raw_query(self, query_feat, query_mask,
                                video_feat1, video_feat2, video_mask,
                                sub_feat1, sub_feat2, sub_mask, cross=False):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat1: (N, Lv, D) or None
            video_feat2:
            video_mask: (N, Lv)
            sub_feat1: (N, Lv, D) or None
            sub_feat2:
            sub_mask: (N, Lv)
            cross:
        """
        video_query, sub_query = self.encode_query(query_feat, query_mask)
        divisor = self.use_sub + self.use_video

        # get video-level retrieval scores
        video_q2ctx_scores = self.get_video_level_scores(video_query, video_feat1, video_mask) if self.use_video else 0
        sub_q2ctx_scores = self.get_video_level_scores(sub_query, sub_feat1, sub_mask) if self.use_sub else 0
        q2ctx_scores = (video_q2ctx_scores + sub_q2ctx_scores) / divisor  # (N, N)

        if self.config.merge_two_stream and self.use_video and self.use_sub:
            st_prob, ed_prob = self.get_merged_st_ed_prob(
                video_query, video_feat2, sub_query, sub_feat2, video_mask, cross=cross)
        else:
            video_st_prob, video_ed_prob = self.get_st_ed_prob(
                video_query, video_feat2, video_mask, module_name="video", cross=cross) if self.use_video else (0, 0)
            sub_st_prob, sub_ed_prob = self.get_st_ed_prob(
                sub_query, sub_feat2, sub_mask, module_name="sub", cross=cross) if self.use_sub else (0, 0)
            st_prob = (video_st_prob + sub_st_prob) / divisor  # (N, Lv)
            ed_prob = (video_ed_prob + sub_ed_prob) / divisor  # (N, Lv)
        return q2ctx_scores, st_prob, ed_prob  # un-normalized masked probabilities!!!!!

    def get_video_level_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """
        bsz = len(query_context_scores)
        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores,
                                                           query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx, loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)
        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)
        sample_min_idx = 1  # skip the masked positive
        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) \
            if self.config.use_hard_negative else bsz
        sampled_neg_score_indices = sorted_scores_indices[
            batch_indices, torch.randint(sample_min_idx, sample_max_idx, size=(bsz,)).to(scores.device)]  # (N, )
        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.config.ranking_loss_type == "hinge":  # max(0, m + S_neg - S_pos)
            return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)
        elif self.config.ranking_loss_type == "lse":  # log[1 + exp(S_neg - S_pos)]
            return torch.log1p(torch.exp(neg_score - pos_score)).sum() / len(pos_score)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)

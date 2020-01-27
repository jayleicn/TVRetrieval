"""
Dataset for clip model
"""
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import math
import random
from utils.basic_utils import load_jsonl, load_json, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from baselines.clip_alignment_with_language.local_utils.proposal import get_proposal_interface
from baselines.clip_alignment_with_language.local_utils.compute_proposal_upper_bound import \
    get_didemo_agreed_ts
from standalone_eval.eval import compute_temporal_iou_batch

logger = logging.getLogger(__name__)


class ProposalRetrievalDataset(Dataset):
    """
    Args:
        dset_name, str, ["tvr"]
        ctx_mode: str,
        pos_iou_thd: float, in [0, 1], >= pos_iou_thd are defined as positive
        neg_iou_thd: float, in [0, 1], < neg_iou_thd are defined as negative
    Return:
        a dict: {
            "meta": {
                "desc_id": int,
                "desc": str,
                "vid_name": str,
                "duration": float,
                "ts": [st (float), ed (float)], seconds, ground_truth timestamps
                "pos_moment": [st (float), ed (float)], seconds, IoU with "ts" >= pos_iou_thd
                "intra_neg_moment": [st (float), ed (float)], seconds, IoU with "ts" < neg_iou_thd
                "inter_neg_vid_name": str,
                "inter_neg_duration": float,
                "inter_neg_moment": [st (float), ed (float)], seconds, IoU with "ts" < neg_iou_thd
            }
            "model_inputs": {
                "desc_feat": torch.tensor, (L, D_t)
                "pos_moment_feat": torch.tensor, (n_clip_in_moment, D)
                "intra_neg_moment_feat": torch.tensor, (n_clip_in_moment, D)
                "inter_neg_moment_feat": torch.tensor, (n_clip_in_moment, D)
            }
        }
    """
    def __init__(self, dset_name, data_path, desc_bert_path, sub_bert_path, max_desc_len,
                 vid_feat_path, clip_length, vid_feat_size, sub_feat_size=0, ctx_mode="video_tef",
                 pos_iou_thd=0.7, neg_iou_thd=0.3, h5driver=None, data_ratio=1.0,
                 normalize_vfeat=True, normalize_tfeat=True, model_type="cal",
                 external_train_vr_res_path=None, video_duration_idx_path=None):
        self.dset_name = dset_name
        self.model_type = model_type
        self.pool_local = model_type == "mcn"  # pool local feature
        self.data_path = data_path
        self.data_ratio = data_ratio

        self.desc_bert_path = desc_bert_path
        self.max_desc_len = max_desc_len
        self.sub_bert_path = sub_bert_path

        self.vid_feat_path = vid_feat_path
        self.clip_length = clip_length
        self.ctx_mode = ctx_mode

        self.pos_iou_thd = pos_iou_thd
        self.neg_iou_thd = neg_iou_thd

        self.vid_feat_output_size = 2 * vid_feat_size * ("video" in ctx_mode) + 2 * ("tef" in ctx_mode)
        self.sub_feat_output_size = 2 * sub_feat_size * ("sub" in ctx_mode) + 2 * ("tef" in ctx_mode)

        # prepare desc data
        self.data = load_jsonl(data_path)
        if self.data_ratio != 1:
            n_examples = int(len(self.data) * data_ratio)
            self.data = self.data[:n_examples]
            logger.info("Using {}% of the data: {} examples".format(data_ratio * 100, n_examples))

        self.proposal_fn = get_proposal_interface(dset_name)
        if self.ctx_mode != "tef":
            self.vid_feat_h5 = h5py.File(self.vid_feat_path, "r", driver=h5driver)
        self.desc_bert_h5 = h5py.File(self.desc_bert_path, "r", driver=h5driver)
        if "sub" in self.ctx_mode:
            self.sub_bert_h5 = h5py.File(self.sub_bert_path, "r", driver=h5driver)
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat
        self.use_video = "video" in self.ctx_mode
        self.use_sub = "sub" in self.ctx_mode
        self.use_tef = "tef" in self.ctx_mode

        if external_train_vr_res_path is not None:
            video_data = load_json(video_duration_idx_path)["train"]
            # {video_idx: [vid_name, vid_duration]}
            video_idx2name_dur_pair = {v[1]: [k, v[0]] for k, v in video_data.items()}
            external_vr_res = load_json(external_train_vr_res_path)
            # {desc_id: [(vid_name, vid_duration), ...]}
            self.desc_id2video_names_dur_pairs = \
                {e["desc_id"]: [video_idx2name_dur_pair[int(sub_e[0])] for sub_e in e["predictions"]]
                 for e in external_vr_res["VR"]}  # ordered

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_data = self.data[index]

        # initialize with basic data
        meta = dict(
            desc_id=raw_data["desc_id"],
            desc=raw_data["desc"],
            vid_name=raw_data["vid_name"],
            duration=raw_data["duration"],
            ts=raw_data["ts"] if self.dset_name != "didemo" else get_didemo_agreed_ts(raw_data["ts"]),
        )
        model_inputs = dict()
        query_feat = self.desc_bert_h5[str(raw_data["desc_id"])][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        model_inputs["query_feat"] = torch.from_numpy(query_feat)

        # sample positive and negative moments
        meta["pos_moment"] = self.align_ts_to_clip_boundaries(meta["duration"], meta["ts"])
        meta["intra_neg_moment"] = self.sample_intra_neg_moment(meta["duration"], meta["ts"])
        meta["inter_neg_moment"], meta["inter_neg_vid_name"], meta["inter_neg_duration"] = \
            self.sample_inter_video_negative(meta["vid_name"], meta["pos_moment"] / meta["duration"],
                                             desc_id=meta["desc_id"])

        pos_tef, intra_neg_tef, inter_neg_tef = (None,) * 3
        if self.use_tef:
            pos_tef = meta["pos_moment"] / meta["duration"]  # temporal endpoint feature, (2, )
            intra_neg_tef = meta["intra_neg_moment"] / meta["duration"]
            inter_neg_tef = meta["inter_neg_moment"] / meta["inter_neg_duration"]

        if self.use_video:
            pos_v_feat = self.vid_feat_h5[meta["vid_name"]]  # (N_frm, D)
            neg_v_feat = self.vid_feat_h5[meta["inter_neg_vid_name"]]
            pos_v_ctx_feat = np.mean(pos_v_feat, axis=0)
            neg_v_ctx_feat = np.mean(neg_v_feat, axis=0)
            if self.normalize_vfeat:
                pos_v_ctx_feat = l2_normalize_np_array(pos_v_ctx_feat)
                neg_v_ctx_feat = l2_normalize_np_array(neg_v_ctx_feat)
            pos_moment_v_feat = self.get_moment_feat(pos_v_feat, meta["pos_moment"],
                                                     normalize=self.normalize_vfeat,
                                                     fix_outbound=True, pool_local=self.pool_local)
            intra_neg_moment_v_feat = self.get_moment_feat(pos_v_feat, meta["intra_neg_moment"],
                                                           normalize=self.normalize_vfeat,
                                                           fix_outbound=True, pool_local=self.pool_local)
            inter_neg_moment_v_feat = self.get_moment_feat(neg_v_feat, meta["inter_neg_moment"],
                                                           normalize=self.normalize_vfeat,
                                                           fix_outbound=True, pool_local=self.pool_local)

            # concat features, [video_clip_feat; video_context_feat; temporal_endpoint_feat]
            model_inputs["pos_moment_video_feat"] = self.concat_feat_adv(
                moment_feats=[pos_moment_v_feat, pos_v_ctx_feat], tef=pos_tef, ctx_mode=self.ctx_mode)
            model_inputs["intra_neg_moment_video_feat"] = self.concat_feat_adv(
                moment_feats=[intra_neg_moment_v_feat, pos_v_ctx_feat], tef=intra_neg_tef, ctx_mode=self.ctx_mode)
            model_inputs["inter_neg_moment_video_feat"] = self.concat_feat_adv(
                moment_feats=[inter_neg_moment_v_feat, neg_v_ctx_feat], tef=inter_neg_tef, ctx_mode=self.ctx_mode)
        else:
            for k in ["pos_moment_video_feat", "intra_neg_moment_video_feat", "inter_neg_moment_video_feat"]:
                model_inputs[k] = torch.zeros((2, 2))

        if self.use_sub:  # no need for ctx feature, as the features are already contextulized
            pos_s_feat = self.sub_bert_h5[meta["vid_name"]]  # (N_words, D_t)
            neg_s_feat = self.sub_bert_h5[meta["inter_neg_vid_name"]]
            pos_s_ctx_feat = np.mean(pos_s_feat, axis=0)
            neg_s_ctx_feat = np.mean(neg_s_feat, axis=0)
            if self.normalize_tfeat:
                pos_s_ctx_feat = l2_normalize_np_array(pos_s_ctx_feat)
                neg_s_ctx_feat = l2_normalize_np_array(neg_s_ctx_feat)
            pos_moment_s_feat = self.get_moment_feat(pos_s_feat, meta["pos_moment"],
                                                     normalize=self.normalize_tfeat,
                                                     fix_outbound=True, pool_local=self.pool_local)
            intra_neg_moment_s_feat = self.get_moment_feat(pos_s_feat, meta["intra_neg_moment"],
                                                           normalize=self.normalize_tfeat,
                                                           fix_outbound=True, pool_local=self.pool_local)
            inter_neg_moment_s_feat = self.get_moment_feat(neg_s_feat, meta["inter_neg_moment"],
                                                           normalize=self.normalize_tfeat,
                                                           fix_outbound=True, pool_local=self.pool_local)

            # concat features, [sub_clip_feat; sub_context_feat; temporal_endpoint_feat]
            model_inputs["pos_moment_sub_feat"] = self.concat_feat_adv(
                moment_feats=[pos_moment_s_feat, pos_s_ctx_feat], tef=pos_tef, ctx_mode=self.ctx_mode)
            model_inputs["intra_neg_moment_sub_feat"] = self.concat_feat_adv(
                moment_feats=[intra_neg_moment_s_feat, pos_s_ctx_feat], tef=intra_neg_tef, ctx_mode=self.ctx_mode)
            model_inputs["inter_neg_moment_sub_feat"] = self.concat_feat_adv(
                moment_feats=[inter_neg_moment_s_feat, neg_s_ctx_feat], tef=inter_neg_tef, ctx_mode=self.ctx_mode)
        else:
            for k in ["pos_moment_sub_feat", "intra_neg_moment_sub_feat", "inter_neg_moment_sub_feat"]:
                model_inputs[k] = torch.zeros((2, 2))

        if not self.use_sub and not self.use_video and self.use_tef:  # use video stream
            model_inputs["pos_moment_video_feat"] = \
                self.concat_feat_adv(tef=pos_tef, ctx_mode=self.ctx_mode)
            model_inputs["intra_neg_moment_video_feat"] = \
                self.concat_feat_adv(tef=intra_neg_tef, ctx_mode=self.ctx_mode)
            model_inputs["inter_neg_moment_video_feat"] = \
                self.concat_feat_adv(tef=inter_neg_tef, ctx_mode=self.ctx_mode)
        return dict(meta=meta, model_inputs=model_inputs)

    def align_ts_to_clip_boundaries(self, duration, ts):
        """  # TODO Do we really need this???
        Generate a moment [st, ed] that is most close to a clip boundary,
        st and ed must be a multiple of self.clip_length, and ed <= duration
        duration: float,
        ts: [st (float), ed (float)], ground_truth ts
        """
        clip_aligned_ts = np.array([math.floor(ts[0] / self.clip_length),
                                    math.ceil(ts[1] / self.clip_length)]) * self.clip_length
        clip_aligned_ts[1] = min(clip_aligned_ts[1], duration)
        return clip_aligned_ts

    def sample_intra_neg_moment(self, duration, ts):
        """ Generate a intra negative moment given the video duration and the GT ts.
        The returned moment will be aligned to clip boundaries.
        1) neg_moment has at least 2 clips
        2) its iou with ts should be < self.neg_iou_thd
        Args:
            duration: float
            ts: [st (float), ed (float)], ground_truth ts

        Returns:

        """
        max_n_search = 5  # search at most max_n_search times, so the program will not be stuck in infinite loops.
        sampled_moments = self.sample_ts_at_clip_boundaries(duration, n_pairs=max_n_search)  # (n_pairs, 2)
        sampled_moments_ious = compute_temporal_iou_batch(sampled_moments, ts)  # (n_pairs, )
        smallest_iou_idx = np.argmin(sampled_moments_ious)
        sampled_moment = sampled_moments[smallest_iou_idx]
        # only a small number (<20 with max_n_search==10) of samples are wrong,
        # usually when the video_duration is too short.
        # if sampled_moments_ious[smallest_iou_idx] >= self.neg_iou_thd:
        #     logger.warning("the sampled intra-neg might be wrong. "
        #                    "v_dur {}, ts {}, sampled neg moment {}, iou {}"
        #                    .format(duration, ts, sampled_moment, sampled_moments_ious[smallest_iou_idx]))
        return sampled_moment

    def sample_ts_at_clip_boundaries(self, duration, n_pairs=1):
        """sample n_pairs moment at clip boundaries, each has at least two clips."""
        # '+ self.clip_length' since we assume indexing using [clip_st_idx, clip_ed_idx),
        moments = np.random.randint(0, np.ceil(duration / self.clip_length), size=(n_pairs, 2))
        moments = np.sort(moments, axis=1) * self.clip_length
        less_equal = moments[:, 1] - moments[:, 0] <= self.clip_length
        start_zero = moments[:, 0] == 0
        moments[:, 1][less_equal * start_zero] += self.clip_length
        moments[:, 0][less_equal * (start_zero == False)] -= self.clip_length  # keep as np.bool!!!
        return moments

    def sample_inter_video_negative(self, pos_vid_name, normalized_pos_moment, desc_id=None):
        """Sample a negative moment --> negative video + similar normalized moment.
        1) they are not from the same video
        Args:
            pos_vid_name: str,
            normalized_pos_moment: np.ndarray, (2, ), value in [0, 1], normalized by duration.
            desc_id: str
        Returns:
            moment: np.ndarray, (2, ), ts aligned to clip boundaries.

        """
        use_guided_negative = hasattr(self, "desc_id2video_names_dur_pairs")
        if use_guided_negative:
            top_videos = self.desc_id2video_names_dur_pairs[desc_id]
            max_idx = len(top_videos) - 1

        while True:  # usually only run once.
            if use_guided_negative:
                sampled_idx = min(max_idx, int(random.expovariate(0.1)))
                sampled_video_name, sampled_video_dur = top_videos[sampled_idx]
            else:
                neg_vid_data = self.data[int(random.random() * len(self))]
                sampled_video_name, sampled_video_dur = neg_vid_data["vid_name"], neg_vid_data["duration"]
            if sampled_video_name != pos_vid_name:
                inter_neg_moment = self.align_ts_to_clip_boundaries(
                    sampled_video_dur, sampled_video_dur * normalized_pos_moment)
                break

        return inter_neg_moment, sampled_video_name, sampled_video_dur

    @classmethod
    def get_clip_indices_from_moments(cls, moment, clip_length):
        clip_st_ed_indices = moment / clip_length
        return math.floor(clip_st_ed_indices[0]), math.ceil(clip_st_ed_indices[1])

    def get_moment_feat(self, vid_feat, moment, normalize=True, fix_outbound=False, pool_local=False):
        """Each moment contains multiple clips.
        Inside means [moment[0], moment[1]] (seconds)
        Args:
            vid_feat: np.ndarray, (N_clips, D)
            moment: [st (float), ed (float)], np.ndarray
            normalize: L2 normalize features
            fix_outbound: bool,
            pool_local: whether to mean pool the features
        Returns:
            moment_feature: np.ndarray, ((moment[1] - moment[0]) / clip_length, D) or (D, )
        """
        clip_st_idx, clip_ed_idx = self.get_clip_indices_from_moments(moment, self.clip_length)
        if fix_outbound:
            vid_feat_len = len(vid_feat)
            if clip_st_idx >= vid_feat_len:
                clip_st_idx = vid_feat_len - 2
        moment_feat = vid_feat[clip_st_idx:clip_ed_idx]  # indexed as [st, ed)
        if pool_local:
            moment_feat = np.mean(moment_feat, axis=0, keepdims=True)
        if normalize:
            moment_feat = l2_normalize_np_array(moment_feat)
        return moment_feat  # (n_clip_in_moment, D) or (D, )

    @classmethod
    def concat_feat_adv(cls, moment_feats=None, tef=None, to_torch=True, ctx_mode="tef"):
        """ Concat moment_feat with other_feats and tef. All the features should be L2 normalized before concatenating
        Args:
            moment_feats: list of feats, one of them might be None. Other possible values are
                ctx_feat (D, ) or sub(vid)_moment_feat (N_p, N_clips, D_t) or (N_clips, D_t).
                The first non-None feature array is used as base for the rest to concatenate with.
            tef: (N_p, 2) or (2, ), np.ndarray
            to_torch: convert resulting np.ndarray to torch.tensor
            ctx_mode:
        """
        if ctx_mode == "tef":
            assembled_feat = np.expand_dims(tef, axis=-2)
        else:  # concat moment_feat with all other_feats
            moment_feats = [e for e in moment_feats if e is not None]  # remove possible None (placeholder)
            extra_dims = moment_feats[0].shape[:-1]  # all others will need to broadcast to match it.
            if isinstance(extra_dims, int):  # happens when len(moment_feat.shape) == 2
                extra_dims = (extra_dims, )
            last_dim_lengths = [0, ] + [e.shape[-1] for e in moment_feats]
            if "tef" in ctx_mode:  # add tef
                last_dim_lengths += [2, ]
                moment_feats += [np.expand_dims(tef, axis=-2), ]

            if len(moment_feats) > 1:
                assembled_feat = np.empty(extra_dims + (sum(last_dim_lengths), ), dtype=np.float32)
                last_dim_lengths_cumsum = [sum(last_dim_lengths[0:idx+1]) for idx in range(len(last_dim_lengths))]
                for idx, feat in enumerate(moment_feats):
                    assembled_feat[..., last_dim_lengths_cumsum[idx]:last_dim_lengths_cumsum[idx+1]] = feat
            else:
                assembled_feat = moment_feats[0]

        if to_torch:
            return torch.from_numpy(assembled_feat)
        else:
            return assembled_feat  # (N_prop, N_clips, D_concat) or (N_clips, D_concat)


class ProposalRetrievalEvalDataset(Dataset):
    """
    init_data_mode: `video_query` or `video_only` or `query_only`,
        it indicates which data to load when initialize the Dataset object.
    data_mode: `context` or `query`, it indicates which data to return for self.__get_item__()
    desc_bert_path_or_handler: h5py.File object or str path
    vid_feat_path_or_handler: h5py.File object or str path
    eval_proposal_bsz: the proposals for a single video will be sorted in length and batched here with
        max batch size to be eval_proposal_bsz. A single video might have multiple batches of proposals.
    load_gt_video: load GroundTruth Video, useful when evaluating single video moment retrieval.
    data_ratio: percentage of query data to use.
    """
    def __init__(self, dset_name, eval_split_name, data_path=None,
                 desc_bert_path_or_handler=None, max_desc_len=None,
                 sub_bert_path_or_handler=None, vid_feat_path_or_handler=None,
                 video_duration_idx_path=None, clip_length=None,
                 eval_proposal_bsz=None, ctx_mode="tef", data_mode="context",
                 h5driver=None, data_ratio=1.0, normalize_vfeat=True,
                 normalize_tfeat=True, max_n_proposals=90, model_type="cal"):
        self.dset_name = dset_name
        self.model_type = model_type
        self.pool_local = model_type == "mcn"  # pool local feature
        self.eval_split_name = eval_split_name
        self.ctx_mode = ctx_mode
        self.load_gt_video = False
        self.data_ratio = data_ratio  # only affect query data
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat
        self.max_n_proposals = max_n_proposals

        self.data_mode = None
        self.set_data_mode(data_mode)

        self.max_desc_len = max_desc_len
        self.data_path = data_path
        self.query_data = load_jsonl(data_path)
        if data_ratio != 1:
            n_examples = int(len(self.query_data) * data_ratio)
            self.query_data = self.query_data[:n_examples]
            logger.info("Using {}% of the data: {} examples".format(data_ratio * 100, n_examples))
        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r", driver=h5driver)

        video_data = load_json(video_duration_idx_path)[self.eval_split_name]
        self.video_data = [{"vid_name": k, "duration": v[0]} for k, v in video_data.items()]
        self.video2idx = {k: v[1] for k, v in video_data.items()}
        self.eval_proposal_bsz = eval_proposal_bsz
        self.clip_length = clip_length
        self.proposal_fn = get_proposal_interface(dset_name)

        self.use_video = "video" in self.ctx_mode
        self.use_sub = "sub" in self.ctx_mode
        self.use_tef = "tef" in self.ctx_mode

        if self.use_video:
            if isinstance(vid_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = vid_feat_path_or_handler
            else:  # str path
                self.vid_feat_h5 = h5py.File(vid_feat_path_or_handler, "r", driver=h5driver)

        if self.use_sub:
            if isinstance(sub_bert_path_or_handler, h5py.File):
                self.sub_bert_h5 = sub_bert_path_or_handler
            else:  # str path
                self.sub_bert_h5 = h5py.File(sub_bert_path_or_handler, "r", driver=h5driver)

    def set_data_mode(self, data_mode):
        """context or query"""
        assert data_mode in ["context", "query"]
        self.data_mode = data_mode

    def load_gt_vid_name_for_query(self, load_gt_video):
        """load_gt_video: bool, affect the returned value of self._get_item_query"""
        assert "vid_name" in self.query_data[0]
        self.load_gt_video = load_gt_video

    def __len__(self):
        if self.data_mode == "context":
            return len(self.video_data)
        else:
            return len(self.query_data)

    def __getitem__(self, index):
        if self.data_mode == "context":
            return self._get_item_context(index)
        else:
            return self._get_item_query(index)

    def _get_item_query(self, index):
        """Need to batch"""
        raw_data = self.query_data[index]

        meta = dict(
            desc_id=raw_data["desc_id"],
            desc=raw_data["desc"],
            vid_name=raw_data["vid_name"] if self.load_gt_video else None
        )

        model_inputs = dict()
        query_feat = self.desc_bert_h5[str(raw_data["desc_id"])][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        model_inputs["query_feat"] = torch.from_numpy(query_feat)
        return dict(meta=meta, model_inputs=model_inputs)

    def _get_item_context(self, index):
        """No need to batch, since it has already been batched here"""
        raw_data = self.video_data[index]

        # get proposals and sort in ascending order, to get more efficient batching
        proposals = self.proposal_fn(
            video_id="", metadata={"duration": raw_data["duration"]})  # np.ndarray (N_p, 2)
        proposals_lengths = proposals[:, 1] - proposals[:, 0]  # seconds
        sorted_proposal_indices = np.argsort(proposals_lengths)[:self.max_n_proposals]
        sorted_proposals = proposals[sorted_proposal_indices]

        # initialize with basic data
        meta = dict(
            vid_name=raw_data["vid_name"],
            duration=raw_data["duration"],
            proposals=sorted_proposals
        )
        model_inputs = dict()

        n_proposal_batches = math.ceil(1.0 * len(sorted_proposals) / self.eval_proposal_bsz)

        tef_batched_list = [None, ] * n_proposal_batches
        t_moments_mask_list = [None, ] * n_proposal_batches
        if self.use_tef:
            tef_array = sorted_proposals / meta["duration"]  # (N_p, 2)
            for batch_idx in range(n_proposal_batches):
                st_m_idx = batch_idx * self.eval_proposal_bsz
                ed_m_idx = (batch_idx + 1) * self.eval_proposal_bsz
                tef_batched_list[batch_idx] = tef_array[st_m_idx:ed_m_idx]
                t_moments_mask_list[batch_idx] = \
                    np.ones((len(tef_batched_list[batch_idx]), 1), dtype=np.float32)
            if not self.use_video and not self.use_sub:  # use video stream
                model_inputs["video_moment_features_list"] = [
                    ProposalRetrievalDataset.concat_feat_adv(tef=t, ctx_mode=self.ctx_mode) for t in tef_batched_list]
                model_inputs["video_moment_mask_list"] = [torch.from_numpy(e) for e in t_moments_mask_list]

        # extract/group/pad
        if self.use_video:
            v_feat = self.vid_feat_h5[meta["vid_name"]]  # (N_frm, D)
            v_ctx_feat = np.mean(v_feat, axis=0)  # (D, )
            if self.normalize_vfeat:
                v_ctx_feat = l2_normalize_np_array(v_ctx_feat)
            v_padded_moments_features_list, v_moments_mask_list = \
                self.get_batched_moment_feat_for_all_proposals(v_feat, sorted_proposals,
                                                               pool_local=self.pool_local,
                                                               normalize=self.normalize_vfeat)

            model_inputs["video_moment_features_list"] = [ProposalRetrievalDataset.concat_feat_adv(
                moment_feats=[v, v_ctx_feat], tef=t, ctx_mode=self.ctx_mode)
                for v, t in zip(v_padded_moments_features_list, tef_batched_list)]
            model_inputs["video_moment_mask_list"] = [torch.from_numpy(e) for e in v_moments_mask_list]

        if self.use_sub:
            s_feat = self.sub_bert_h5[meta["vid_name"]]  # (N_frm, D)
            s_ctx_feat = np.mean(s_feat, axis=0)  # (D, )
            if self.normalize_tfeat:
                s_ctx_feat = l2_normalize_np_array(s_ctx_feat)
            s_padded_moments_features_list, s_moments_mask_list = \
                self.get_batched_moment_feat_for_all_proposals(s_feat, sorted_proposals,
                                                               pool_local=self.pool_local,
                                                               normalize=self.normalize_tfeat)
            model_inputs["sub_moment_features_list"] = [ProposalRetrievalDataset.concat_feat_adv(
                moment_feats=[s, s_ctx_feat], tef=t, ctx_mode=self.ctx_mode)
                for s, t in zip(s_padded_moments_features_list, tef_batched_list)]
            model_inputs["sub_moment_mask_list"] = [torch.from_numpy(e) for e in s_moments_mask_list]
        return dict(meta=meta, model_inputs=model_inputs)

    def get_batched_moment_feat_for_all_proposals(self, feature, moments, pool_local=False, normalize=True):
        """proposals of the same video wil be segmented into multiple batches to accomodate GPU memory
        pool_local: pool local feature into a single vector
        """
        n_proposal_batches = math.ceil(1.0 * len(moments) / self.eval_proposal_bsz)
        padded_moments_features_list = [None, ] * n_proposal_batches
        moments_mask_list = [None, ] * n_proposal_batches
        moments_features = self.get_moment_feat_for_all_proposals(
            feature, moments, normalize=normalize, pool_local=pool_local)  # N_p * [(N_clips, D), ]
        for batch_idx in range(n_proposal_batches):
            st_m_idx = batch_idx * self.eval_proposal_bsz
            ed_m_idx = (batch_idx + 1) * self.eval_proposal_bsz
            padded_moments_features, moments_mask = \
                pad_sequences_1d(moments_features[st_m_idx:ed_m_idx], dtype=np.float32)
            padded_moments_features_list[batch_idx] = padded_moments_features
            moments_mask_list[batch_idx] = moments_mask
            assert np.sum(np.sum(moments_mask, axis=1) == 0) == 0, " err {}".format(moments_mask)
        assert np.sum(np.sum(moments_mask_list[0], axis=1) == 0) == 0, " err {}".format(moments_mask_list)
        return padded_moments_features_list, moments_mask_list

    def get_moment_feat_for_all_proposals(self, vid_feat, moments, normalize=True, pool_local=False):
        """Each moment is comprised of multiple clips
        Args:
            vid_feat: np.ndarray, (N_clips, D)
            moments: np.ndarray, (N_p, 2), each row is [st (float), ed (float)],
            normalize: L2 normalize
            pool_local:
        Returns:
            moments_features: list(np.ndarray), [(N_clips, D), ] * N_p, N_clips is changing.
        """
        if normalize and not pool_local:
            vid_feat = l2_normalize_np_array(vid_feat)
        vid_feat_len = len(vid_feat)
        moments_st_clip_indices = np.floor(moments[:, 0] / self.clip_length).astype(np.int).clip(0, vid_feat_len-1)
        moments_ed_clip_indices = np.ceil(moments[:, 1] / self.clip_length).astype(np.int).clip(1, vid_feat_len)
        moments_features = []
        for st_idx, ed_idx, m in zip(moments_st_clip_indices, moments_ed_clip_indices, moments):
            feat = vid_feat[st_idx:ed_idx]
            if pool_local:
                feat = np.mean(feat, axis=0, keepdims=True)
                if normalize:
                    feat = l2_normalize_np_array(feat)
            moments_features.append(feat)
        return moments_features


def proposal_retrieval_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = {k: pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=torch.float32)
                    for k in model_inputs_keys}
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = {}
    for k, v in batched_model_inputs.items():
        model_inputs[k] = v[0].to(device, non_blocking=non_blocking)
        model_inputs[k.replace("feat", "mask")] = v[1].to(device, non_blocking=non_blocking)
    return model_inputs


if __name__ == '__main__':
    from baselines.clip_alignment_with_language.config import BaseOptions
    options = BaseOptions().parse()

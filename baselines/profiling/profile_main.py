"""
Profile the time needed for retrieval.
We consider retrieval in a corpus of 1M videos, 1K videos are added, 10K queries are retrieved.
Calculate the time needed for adding 1K videos, and performing retrieval for 10K queries.

1, Data Loading time is ignored, consider it is hidden by computation time.
2, Sort time is ignored, since it is the similar among the methods.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pprint
from tqdm import tqdm, trange
from baselines.crossmodal_moment_localization.model_xml import XML, xml_base_config
from baselines.mixture_embedding_experts.model import MEE, mee_base_cfg
from baselines.clip_alignment_with_language.model import CALWithSub, cal_base_cfg
from baselines.excl.model import EXCL, excl_base_cfg
from utils.basic_utils import save_json


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)


class ProfileBase(object):
    N_NewQuery = 1e4
    N_NewVideo = 1e3
    N_Videos = 1e6
    AvgVideoLength = 100
    ClipLength = 5
    AvgClipPerVideo = int(AvgVideoLength / ClipLength)  # max_ctx_l
    AvgWordInQuery = 15
    # estimated by
    # scales=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  => max_proposal = 14
    AvgProposalPerVideo = 170
    MaxClipPerProposal = 14  # pad to this length
    AvgClipPerProposal = 7  # 6.88

    VideoFeatureDim = 3074  # 1024 + 2048 + 2 (TEF)
    SubFeatureDim = 770
    QueryFeatureDim = 768

    HiddenSize = 256
    N_Runs = 5  # Get the average time

    def __init__(self, device=torch.device("cuda:0"), ctx_batch_size=400, query_batch_size=100):
        self.device = device
        self.ctx_batch_size = ctx_batch_size
        self.query_batch_size = query_batch_size
        self.model_config = self.get_model_config()
        print(self.model_config)
        self.model = self.get_model()

    def get_model(self):
        return None

    def get_model_config(self):
        return None

    def set_ctx_batch_size(self, batch_size):
        self.ctx_batch_size = batch_size

    def set_query_batch_size(self, batch_size):
        self.query_batch_size = batch_size

    def cast_dict_inputs_to_device(self, dict_inputs, device):
        return {k: v.to(device) for k, v in dict_inputs.items()}

    def get_fake_ctx_raw_input_st_ed(self, no_tef=False):
        return dict(
            video_feat=torch.FloatTensor(self.ctx_batch_size, self.model_config.max_ctx_l,
                                         self.VideoFeatureDim - 2*no_tef),
            sub_feat=torch.FloatTensor(self.ctx_batch_size, self.model_config.max_ctx_l, self.SubFeatureDim - 2*no_tef),
            ctx_mask=torch.FloatTensor(self.ctx_batch_size, self.model_config.max_ctx_l),
        )

    def get_fake_raw_query(self):
        return dict(
            query_feat=torch.FloatTensor(self.query_batch_size, self.AvgWordInQuery, self.QueryFeatureDim),
            query_mask=torch.ones(self.query_batch_size, self.AvgWordInQuery)
        )


"""
from baselines.profiling.profile_main import ProfileXML
profile_xml = ProfileXML(ctx_batch_size=400, query_batch_size=100)
profile_xml.get_ctx_encoding_time()
"""


class ProfileXML(ProfileBase):
    def get_model_config(self):
        xml_base_config["ctx_mode"] = "video_sub_tef"
        xml_base_config["merge_two_stream"] = True
        xml_base_config["cross_att"] = True
        xml_base_config["max_ctx_l"] = self.AvgClipPerVideo
        xml_base_config["visual_input_size"] = self.VideoFeatureDim
        xml_base_config["query_input_size"] = self.QueryFeatureDim
        xml_base_config["sub_input_size"] = self.SubFeatureDim
        xml_base_config["hidden_size"] = self.HiddenSize
        return xml_base_config

    def get_model(self):
        model = XML(self.model_config)
        model.to(self.device)
        model.eval()
        return model

    def get_fake_encoded_ctx(self):
        return dict(
            ctx_feat=torch.FloatTensor(self.ctx_batch_size, self.model_config.max_ctx_l, self.HiddenSize),
            ctx_mask=torch.FloatTensor(self.ctx_batch_size, self.model_config.max_ctx_l),
        )

    def get_fake_encoded_query(self):
        return dict(query_feat=torch.FloatTensor(self.ctx_batch_size, self.HiddenSize))

    def _get_ctx_encoding_time(self, video_feat, sub_feat, ctx_mask):
        """Considered two modalities"""
        torch.cuda.synchronize()
        st_time = time.time()
        self.model.cross_encode_context(video_feat, ctx_mask, sub_feat, ctx_mask)
        torch.cuda.synchronize()
        return time.time() - st_time

    def get_ctx_encoding_time(self):
        with torch.no_grad():
            fake_ctx_inputs = self.cast_dict_inputs_to_device(self.get_fake_ctx_raw_input_st_ed(), self.device)
            raw_video = fake_ctx_inputs["video_feat"]
            raw_sub = fake_ctx_inputs["sub_feat"]
            ctx_mask = fake_ctx_inputs["ctx_mask"]
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_ctx_encoding_time(raw_video, raw_sub, ctx_mask)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))

    def _get_query_encoding_time(self, raw_query, query_mask):
        """Considered two modalities"""
        torch.cuda.synchronize()
        st_time = time.time()
        encoded_query = self.model.encode_input(raw_query, query_mask,
                                                self.model.query_input_proj,
                                                self.model.query_encoder,
                                                self.model.query_pos_embed)  # (N, Lq, D)
        # video level
        video_query, sub_query = \
            self.model.get_modularized_queries(encoded_query, query_mask, return_modular_att=False)
        # st ed
        video_query = self.model.video_query_linear(video_query)
        sub_query = self.model.sub_query_linear(sub_query)
        torch.cuda.synchronize()
        return time.time() - st_time

    def get_query_encoding_time(self):
        with torch.no_grad():
            query_inputs = self.cast_dict_inputs_to_device(self.get_fake_raw_query(), self.device)
            raw_query = query_inputs["query_feat"]
            query_mask = query_inputs["query_mask"]
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_query_encoding_time(raw_query, query_mask)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))

    def _get_retrieval_time(self, encoded_video_query, encoded_video, ctx_mask):
        """Consider the queries are encoded, Calculate in a single modality then multiply by 2."""
        torch.cuda.synchronize()
        st_time = time.time()
        self.model.get_video_level_scores(encoded_video_query, encoded_video, ctx_mask)
        torch.cuda.synchronize()
        return (time.time() - st_time) * 2

    def get_retrieval_time(self):
        with torch.no_grad():
            encoded_query = self.cast_dict_inputs_to_device(self.get_fake_encoded_query(), self.device)["query_feat"]
            fake_ctx_inputs = self.cast_dict_inputs_to_device(self.get_fake_encoded_ctx(), self.device)
            encoded_ctx = fake_ctx_inputs["ctx_feat"]
            ctx_mask = fake_ctx_inputs["ctx_mask"]
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_retrieval_time(encoded_query, encoded_ctx, ctx_mask)]
            times = torch.FloatTensor(times)  # since we have two modalities
        return dict(avg=float(times.mean()), std=float(times.std()))

    def _get_span_prediction_time(self, query_feat, ctx_feat, ctx_mask):
        """Considered two modalities"""
        torch.cuda.synchronize()
        st_time = time.time()
        similarity = torch.einsum("md,nld->mnl", query_feat, ctx_feat)
        similarity = (similarity + similarity) / 2  # (Nq, Nv, L)  from query to all videos.
        n_q, n_c, l = similarity.shape
        similarity = similarity.view(n_q * n_c, 1, l)
        st_prob = self.model.merged_st_predictor(similarity).view(n_q, n_c, l)  # (Nq, Nv, L)
        ed_prob = self.model.merged_ed_predictor(similarity).view(n_q, n_c, l)  # (Nq, Nv, L)
        st_prob = mask_logits(st_prob, ctx_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, ctx_mask)
        torch.cuda.synchronize()
        return time.time() - st_time

    def get_span_prediction_time(self):
        with torch.no_grad():
            encoded_query = self.cast_dict_inputs_to_device(self.get_fake_encoded_query(), self.device)["query_feat"]
            fake_ctx_inputs = self.cast_dict_inputs_to_device(self.get_fake_encoded_ctx(), self.device)
            encoded_ctx = fake_ctx_inputs["ctx_feat"]
            ctx_mask = fake_ctx_inputs["ctx_mask"]
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_span_prediction_time(encoded_query, encoded_ctx, ctx_mask)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))


"""
from baselines.profiling.profile_main import ProfileMEE
profile_mee = ProfileMEE(ctx_batch_size=400, query_batch_size=100)
profile_mee.get_ctx_encoding_time()
"""


class ProfileMEE(ProfileBase):
    def get_model_config(self):
        mee_base_cfg["ctx_mode"] = "video_sub"
        mee_base_cfg["text_input_size"] = self.QueryFeatureDim
        mee_base_cfg["vid_input_size"] = self.VideoFeatureDim
        mee_base_cfg["output_size"] = self.HiddenSize
        return mee_base_cfg

    def get_model(self):
        model = MEE(self.model_config)
        model.to(self.device)
        model.eval()
        return model

    def get_fake_raw_ctx(self):
        return dict(
            vid_feat=torch.FloatTensor(self.ctx_batch_size, self.VideoFeatureDim),
            sub_feat=torch.FloatTensor(self.ctx_batch_size, self.QueryFeatureDim)
        )

    def get_fake_encoded_ctx_query(self):
        return dict(
            ctx_feat=torch.FloatTensor(self.ctx_batch_size, self.HiddenSize),
            query_feat=torch.FloatTensor(self.ctx_batch_size, self.HiddenSize)
        )

    def _get_ctx_encoding_time(self, vid_feat, sub_feat):
        torch.cuda.synchronize()
        st_time = time.time()
        self.model.video_gu(vid_feat)
        self.model.sub_gu(sub_feat)
        torch.cuda.synchronize()
        return time.time() - st_time

    def get_ctx_encoding_time(self):
        feat_dict = self.cast_dict_inputs_to_device(self.get_fake_raw_ctx(), self.device)
        with torch.no_grad():
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_ctx_encoding_time(**feat_dict)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))

    def _get_query_encoding_time(self, query_feat):
        """Considered 2 modalities"""
        torch.cuda.synchronize()
        st_time = time.time()
        pooled_query = self.model.query_pooling(query_feat)  # (N, Dt)
        video_query = self.model.video_query_gu(pooled_query)
        sub_query = self.model.sub_query_gu(pooled_query)
        stream_weights = self.model.moe_fc(pooled_query)  # (N, 2)
        torch.cuda.synchronize()
        return time.time() - st_time

    def get_query_encoding_time(self):
        raw_query = self.cast_dict_inputs_to_device(self.get_fake_raw_query(), self.device)["query_feat"]
        with torch.no_grad():
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_query_encoding_time(raw_query)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))

    def _get_retrieval_time(self, encoded_query, encoded_ctx):
        """Considered 2 modalities"""
        torch.cuda.synchronize()
        st_time = time.time()
        torch.einsum("md,nd->mn", encoded_query, encoded_ctx)  # (N, N)
        torch.cuda.synchronize()
        return (time.time() - st_time) * 2

    def get_retrieval_time(self):
        model_inputs = self.cast_dict_inputs_to_device(self.get_fake_encoded_ctx_query(), self.device)
        encoded_query = model_inputs["ctx_feat"]
        encoded_ctx = model_inputs["query_feat"]
        with torch.no_grad():
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_retrieval_time(encoded_query, encoded_ctx)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))


class ProfileCAL(ProfileBase):
    def get_model_config(self):
        cal_base_cfg["ctx_mode"] = "video_sub"
        cal_base_cfg["embedding_size"] = self.QueryFeatureDim
        cal_base_cfg["visual_input_size"] = self.VideoFeatureDim * 2
        cal_base_cfg["textual_input_size"] = self.SubFeatureDim * 2
        cal_base_cfg["output_size"] = self.HiddenSize
        return cal_base_cfg

    def get_model(self):
        model = CALWithSub(self.model_config)
        model.to(self.device)
        model.eval()
        return model

    def get_fake_raw_ctx(self, model_name="cal"):
        """The features are `*2` since they use both global and local features"""
        return dict(
            sub_feat=torch.FloatTensor(self.ctx_batch_size, self.AvgProposalPerVideo,
                                       self.AvgClipPerProposal, self.SubFeatureDim * 2),
            vid_feat=torch.FloatTensor(self.ctx_batch_size, self.AvgProposalPerVideo,
                                       self.AvgClipPerProposal, self.VideoFeatureDim * 2))

    def _get_ctx_encoding_time(self, sub_feat, vid_feat, model_name="cal"):
        if model_name == "mcn":
            sub_feat = sub_feat.sum(2)
            vid_feat = vid_feat.sum(2)
        torch.cuda.synchronize()
        st_time = time.time()
        self.model.moment_encoder(vid_feat, module_name="video")
        self.model.moment_encoder(sub_feat, module_name="sub")
        torch.cuda.synchronize()
        return time.time() - st_time

    def get_ctx_encoding_time(self, model_name="cal"):
        """model_name: str, `cal` or `mcn`"""
        feat_dict = self.cast_dict_inputs_to_device(
            self.get_fake_raw_ctx(model_name=model_name), self.device)
        feat_dict["model_name"] = model_name
        with torch.no_grad():
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_ctx_encoding_time(**feat_dict)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))

    def _get_query_encoding_time(self, query_feat, query_mask):
        torch.cuda.synchronize()
        st_time = time.time()
        self.model.query_encoder(query_feat, query_mask)
        torch.cuda.synchronize()
        return time.time() - st_time

    def get_query_encoding_time(self):
        feat_dict = self.cast_dict_inputs_to_device(self.get_fake_raw_query(), self.device)
        with torch.no_grad():
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_query_encoding_time(**feat_dict)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))


class ProfileExCL(ProfileBase):
    def get_model_config(self):
        excl_base_cfg["ctx_mode"] = "video_sub"
        excl_base_cfg["query_input_size"] = self.QueryFeatureDim
        excl_base_cfg["visual_input_size"] = self.VideoFeatureDim
        excl_base_cfg["sub_input_size"] = self.SubFeatureDim
        excl_base_cfg["output_size"] = self.HiddenSize
        return excl_base_cfg

    def get_model(self):
        model = EXCL(self.model_config)
        model.to(self.device)
        model.eval()
        return model

    def get_fake_raw_input(self):
        """The features are `*2` since they use both global and local features"""
        return dict(
            query_feat=torch.FloatTensor(self.ctx_batch_size, self.AvgWordInQuery, self.QueryFeatureDim),
            query_mask=torch.ones((self.ctx_batch_size, self.AvgWordInQuery)),
            sub_feat=torch.FloatTensor(self.ctx_batch_size,  self.AvgClipPerVideo, self.SubFeatureDim),
            sub_mask=torch.ones(self.ctx_batch_size,  self.AvgClipPerVideo),
            video_feat=torch.FloatTensor(self.ctx_batch_size,  self.AvgClipPerVideo, self.VideoFeatureDim),
            video_mask=torch.ones(self.ctx_batch_size,  self.AvgClipPerVideo),
            tef_feat=torch.FloatTensor(self.ctx_batch_size,  self.AvgClipPerVideo, 2),
            tef_mask=torch.ones(self.ctx_batch_size,  self.AvgClipPerVideo),
            st_ed_indices=torch.ones(2, 2),  # not used.
        )

    def _get_prediction_time(self, input_dict):
        torch.cuda.synchronize()
        st_time = time.time()
        self.model(**input_dict)
        torch.cuda.synchronize()
        return time.time() - st_time

    def get_prediction_time(self):
        """model_name: str, `cal` or `mcn`"""
        feat_dict = self.cast_dict_inputs_to_device(
            self.get_fake_raw_input(), self.device)
        feat_dict["is_training"] = False
        with torch.no_grad():
            times = []
            for _ in trange(self.N_Runs):
                times += [self._get_prediction_time(feat_dict)]
            times = torch.FloatTensor(times)
        return dict(avg=float(times.mean()), std=float(times.std()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="")
    parser.add_argument("--ctx_batch_size", type=int, default=400)
    parser.add_argument("--query_batch_size", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="baselines/profiling/cache")
    args = parser.parse_args()

    model = args.model
    query_batch_size = args.query_batch_size
    ctx_batch_size = args.ctx_batch_size
    if model == "mee":
        profile_mee = ProfileMEE(ctx_batch_size=ctx_batch_size, query_batch_size=query_batch_size)
        # use the 2nd one to report time
        profile_mee.get_ctx_encoding_time()
        ctx_enc_time = profile_mee.get_ctx_encoding_time()
        query_enc_time = profile_mee.get_query_encoding_time()
    elif model == "cal":
        profile_cal = ProfileCAL(ctx_batch_size=ctx_batch_size, query_batch_size=query_batch_size)
        # use the 2nd one to report time
        profile_cal.get_ctx_encoding_time()
        ctx_enc_time = profile_cal.get_ctx_encoding_time(model_name="cal")
        query_enc_time = profile_cal.get_query_encoding_time()
    elif model == "mcn":
        profile_cal = ProfileCAL(ctx_batch_size=ctx_batch_size, query_batch_size=query_batch_size)
        # use the 2nd one to report time
        profile_cal.get_ctx_encoding_time()
        ctx_enc_time = profile_cal.get_ctx_encoding_time(model_name="mcn")
        query_enc_time = profile_cal.get_query_encoding_time()
    elif model == "xml":
        profile_xml = ProfileXML(ctx_batch_size=ctx_batch_size, query_batch_size=query_batch_size)
        # use the 2nd one to report time
        profile_xml.get_ctx_encoding_time()
        ctx_enc_time = profile_xml.get_ctx_encoding_time()
        query_enc_time = profile_xml.get_query_encoding_time()
    elif model == "excl":
        profile_excl = ProfileExCL(ctx_batch_size=ctx_batch_size, query_batch_size=ctx_batch_size)
        # use the 2nd one to report time
        profile_excl.get_prediction_time()
        ctx_enc_time = profile_excl.get_prediction_time()
        query_enc_time = 0
        # Calculate the total time as ctx_enc_time * (100 * 1M / ctx_batch_size)
    else:
        raise NotImplementedError
    # ctx_enc_time = ctx_enc_time
    save_path = os.path.join(args.save_dir, "{}_profile_main.json".format(model))

    n_videos = ProfileBase.N_Videos
    res = dict(
        ctx_enc_time=ctx_enc_time,
        ctx_enc_avg_time_all_videos=ctx_enc_time["avg"] * n_videos / ctx_batch_size,
        query_enc_time=query_enc_time,
        n_videos=n_videos,
        ctx_batch_size=ctx_batch_size,
        query_batch_size=query_batch_size,
        model=model
    )
    save_json(res, save_path, save_pretty=True)
    pprint.pprint(res)

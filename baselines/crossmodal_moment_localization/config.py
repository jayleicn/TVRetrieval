import os
import time
import torch
import argparse

from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile
from baselines.clip_alignment_with_language.local_utils.proposal import ProposalConfigs


class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--dset_name", type=str, choices=["tvr"])
        self.parser.add_argument("--eval_split_name", type=str, default="val",
                                 help="should match keys in video_duration_idx_path, must set for VCMR")
        self.parser.add_argument("--debug", action="store_true",
                                 help="debug (fast) mode, break all loops, do not load all data into memory.")
        self.parser.add_argument("--data_ratio", type=float, default=1.0,
                                 help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                      "Use small portion for debug purposes. Note this is different from --debug, "
                                      "which works by breaking the loops, typically they are not used together.")
        self.parser.add_argument("--results_root", type=str, default="results")
        self.parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        self.parser.add_argument("--seed", type=int, default=2018, help="random seed")
        self.parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument("--num_workers", type=int, default=8,
                                 help="num subprocesses used to load the data, 0: use main process")
        self.parser.add_argument("--no_core_driver", action="store_true",
                                 help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")
        self.parser.add_argument("--no_pin_memory", action="store_true",
                                 help="Don't use pin_memory=True for dataloader. "
                                      "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # training config
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_warmup_proportion", type=float, default=0.01,
                                 help="Proportion of training to perform linear learning rate warmup for. "
                                      "E.g., 0.1 = 10% of training.")
        self.parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=10,
                                 help="number of epochs to early stop, use -1 to disable early stop")
        self.parser.add_argument("--stop_task", type=str, default="VCMR", choices=["VCMR", "SVMR", "VR"],
                                 help="Use metric associated with stop_task for early stop")
        self.parser.add_argument("--eval_tasks_at_training", type=str, nargs="+",
                                 default=["VCMR", "SVMR", "VR"], choices=["VCMR", "SVMR", "VR"],
                                 help="evaluate and report  numbers for tasks specified here.")
        self.parser.add_argument("--bsz", type=int, default=128, help="mini-batch size")
        self.parser.add_argument("--eval_query_bsz", type=int, default=50,
                                 help="mini-batch size at inference, for query")
        self.parser.add_argument("--eval_context_bsz", type=int, default=200,
                                 help="mini-batch size at inference, for video/sub")
        self.parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        self.parser.add_argument("--grad_clip", type=float, default=-1, help="perform gradient clip, -1: disable")
        self.parser.add_argument("--margin", type=float, default=0.1, help="margin for   hinge loss")
        self.parser.add_argument("--lw_neg_q", type=float, default=1,
                                 help="weight for ranking loss with negative query and positive context")
        self.parser.add_argument("--lw_neg_ctx", type=float, default=1,
                                 help="weight for ranking loss with positive query and negative context")
        self.parser.add_argument("--lw_st_ed", type=float, default=0.01, help="weight for st ed prediction loss")
        self.parser.add_argument("--train_span_start_epoch", type=int, default=0,
                                 help="which epoch to start training span prediction, -1 to disable")
        self.parser.add_argument("--ranking_loss_type", type=str, default="hinge", choices=["hinge", "lse"],
                                 help="att loss type, can be hinge loss or its smooth approximation LogSumExp")
        self.parser.add_argument("--hard_negtiave_start_epoch", type=int, default=20,
                                 help="which epoch to start hard negative sampling for video-level ranking loss,"
                                      "use -1 to disable")
        self.parser.add_argument("--hard_pool_size", type=int, default=20,
                                 help="hard negatives are still sampled, but from a harder pool.")

        # Model and Data config
        self.parser.add_argument("--max_sub_l", type=int, default=50,
                                 help="max length of all sub sentence 97.71 under 50 for 3 sentences")
        self.parser.add_argument("--max_desc_l", type=int, default=30, help="max length of descriptions")
        self.parser.add_argument("--max_ctx_l", type=int, default=100,
                                 help="max number of snippets, 100 for tvr clip_length=1.5, oly 109/21825 > 100")

        self.parser.add_argument("--train_path", type=str, default=None)
        self.parser.add_argument("--eval_path", type=str, default=None,
                                 help="Evaluating during training, for Dev set. If None, will only do training, "
                                      "anet_cap and charades_sta has no dev set, so None")
        self.parser.add_argument("--external_inference_vr_res_path", type=str, default=None,
                                 help="if set, use external video retrieval results to guide evaluation. ")
        self.parser.add_argument("--use_glove", action="store_true", help="Use GloVe instead of BERT features")
        self.parser.add_argument("--word2idx_path", type=str,
                                 help="a dict, {word: word_idx, ...}, "
                                      "special tokens are {<pad>: 0, <unk>: 1, <eos>: 2}")
        self.parser.add_argument("--vocab_size", type=int, default=-1,
                                 help="Set automatically to len(word2idx)")
        self.parser.add_argument("--glove_path", type=str,
                                 help="path to file containing the GloVe embeddings for words in word2idx")
        self.parser.add_argument("--desc_bert_path", type=str, default=None)
        self.parser.add_argument("--sub_bert_path", type=str, default=None)
        self.parser.add_argument("--sub_feat_size", type=int, default=768, help="feature dim for sub feature")
        self.parser.add_argument("--q_feat_size", type=int, default=768, help="feature dim for sub feature")
        self.parser.add_argument("--ctx_mode", type=str, choices=["video", "sub", "video_sub", "tef",
                                                                  "video_tef", "sub_tef", "video_sub_tef"],
                                 help="which context to use. a combination of [video, sub, tef]")
        self.parser.add_argument("--video_duration_idx_path", type=str, default=None)
        self.parser.add_argument("--vid_feat_path", type=str, default="")
        self.parser.add_argument("--no_norm_vfeat", action="store_true",
                                 help="Do not do normalization on video feat, use it only when using resnet_i3d feat")
        self.parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalization on text feat")
        self.parser.add_argument("--clip_length", type=float, default=None,
                                 help="each video will be uniformly segmented into small clips, "
                                      "will automatically loaded from ProposalConfigs if None")
        self.parser.add_argument("--vid_feat_size", type=int, help="feature dim for video feature")

        self.parser.add_argument("--span_predictor_type", type=str, default="conv", choices=["conv", "cat_linear"],
                                 help="how to generate span predictions, "
                                      "conv: apply 1D-Conv layer on top of NxL dot product of query and clips"
                                      "cat_linear: cat the query and clips then use a linear layer to give output. "
                                      "Note cat_linear is implemented as first project query and clips into scores, "
                                      "separately, then sum them up, this should be similar to first cat then project.")
        self.parser.add_argument("--stack_conv_predictor_conv_kernel_sizes", type=int, default=-1, nargs="+",
                                 help="combine the results from conv edge detectors of all sizes specified."
                                      "-1: disable. If specified, will ignore --conv_kernel_size option."
                                      "This flag is only used when --merge_two_stream and --span_predictor_type conv!")
        self.parser.add_argument("--encoder_type", type=str, default="transformer",
                                 choices=["gru", "lstm", "transformer", "cnn"])
        self.parser.add_argument("--add_pe_rnn", action="store_true",
                                 help="Add positional encoding for GRU and LSTM encoder as well")
        self.parser.add_argument("--no_merge_two_stream", action="store_true", help="do not merge video and subtitles")
        self.parser.add_argument("--no_cross_att", action="store_true",
                                 help="Use cross-attention for modeling video and subtitles")
        self.parser.add_argument("--no_self_att", action="store_true", help="do not use self attention")
        self.parser.add_argument("--no_modular", action="store_true", help="do not use modular attention")
        self.parser.add_argument("--pe_type", type=str, default="cosine", choices=["none", "linear", "cosine"],
                                 help="Only for query encoding")
        self.parser.add_argument("--max_position_embeddings", type=int, default=300)
        self.parser.add_argument("--hidden_size", type=int, default=256)
        self.parser.add_argument("--n_heads", type=int, default=4)
        self.parser.add_argument("--input_drop", type=float, default=0.1, help="Applied to all inputs")
        self.parser.add_argument("--drop", type=float, default=0.1, help="Applied to all other layers")
        self.parser.add_argument("--cross_att_drop", type=float, default=0.1, help="Applied to cross-att")
        self.parser.add_argument("--conv_kernel_size", type=int, default=5)
        self.parser.add_argument("--conv_stride", type=int, default=1)
        self.parser.add_argument("--initializer_range", type=float, default=0.02,
                                 help="initializer range for linear layer")

        # post processing
        self.parser.add_argument("--min_pred_l", type=int, default=2,
                                 help="constrain the [st, ed] with ed - st >= 2"
                                      "(2 clips with length 1.5 each, 3 secs in total"
                                      "this is the min length for proposal-based method)")
        self.parser.add_argument("--max_pred_l", type=int, default=16,
                                 help="constrain the [st, ed] pairs with ed - st <= 16, 24 secs in total"
                                      "(16 clips with length 1.5 each, "
                                      "this is the max length for proposal-based method)")
        self.parser.add_argument("--q2c_alpha", type=float, default=20,
                                 help="give more importance to top scored videos' spans,  "
                                      "the new score will be: s_new = exp(alpha * s), "
                                      "higher alpha indicates more importance. Note s in [-1, 1]")

        self.parser.add_argument("--max_before_nms", type=int, default=200)
        self.parser.add_argument("--max_vcmr_video", type=int, default=100,
                                 help="re-ranking in top-max_vcmr_video")
        self.parser.add_argument("--nms_thd", type=float, default=-1,
                                 help="additionally use non-maximum suppression "
                                      "(or non-minimum suppression for distance)"
                                      "to post-processing the predictions. "
                                      "-1: do not use nms. 0.6 for charades_sta, 0.5 for anet_cap,")

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print("------------ Options -------------\n{}\n-------------------"
              .format({str(k): str(v) for k, v in sorted(args.items())}))

        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.no_core_driver = True
            opt.num_workers = 0
            opt.eval_query_bsz = 100

        if isinstance(self, TestOptions):
            # modify model_dir to absolute path
            opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "num_workers", "nms_thd", "debug",
                               "eval_split_name", "eval_path", "eval_query_bsz", "eval_context_bsz",
                               "max_pred_l", "min_pred_l", "external_inference_vr_res_path"]:
                    setattr(opt, arg, saved_options[arg])
            # opt.no_core_driver = True
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            if opt.clip_length is None:
                opt.clip_length = ProposalConfigs[opt.dset_name]["clip_length"]
                print("Loaded clip_length {} from proposal config file".format(opt.clip_length))
            opt.results_dir = os.path.join(opt.results_root,
                                           "-".join([opt.dset_name, opt.ctx_mode, opt.exp_id,
                                                     time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)
            # save a copy of current code
            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code",
                         exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"],)

        self.display_save(opt)

        if "sub" in opt.ctx_mode:
            assert opt.dset_name == "tvr", "sub is only supported for tvr dataset"

        if opt.hard_negtiave_start_epoch != -1:
            if opt.hard_pool_size > opt.bsz:
                print("[WARNING] hard_pool_size is larger than bsz")

        assert opt.stop_task in opt.eval_tasks_at_training
        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")
        opt.h5driver = None if opt.no_core_driver else "core"
        # num_workers > 1 will only work with "core" mode, i.e., memory-mapped hdf5
        opt.num_workers = 1 if opt.no_core_driver else opt.num_workers
        opt.pin_memory = not opt.no_pin_memory

        if "video" in opt.ctx_mode and opt.vid_feat_size > 3000:  # 3072, the normalized concatenation of resnet+i3d
            assert opt.no_norm_vfeat

        if "tef" in opt.ctx_mode and "video" in opt.ctx_mode:
            opt.vid_feat_size += 2
        if "tef" in opt.ctx_mode and "sub" in opt.ctx_mode:
            opt.sub_feat_size += 2

        if "video" not in opt.ctx_mode or "sub" not in opt.ctx_mode:
            opt.no_merge_two_stream = True
            opt.no_cross_att = True

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
        self.parser.add_argument("--tasks", type=str, nargs="+",
                                 choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"],
                                 help="Which tasks to run."
                                      "VCMR: Video Corpus Moment Retrieval;"
                                      "SVMR: Single Video Moment Retrieval;"
                                      "VR: regular Video Retrieval. (will be performed automatically with VCMR)")

"""
Finetune on datasets with single sentences as inputs, using MLM objective.

This script has been verified in the following framework
$ git clone https://github.com/huggingface/transformers.git
$ cd transformers
$ git checkout e1b2949ae6cb34cc39e3934ca87423474f8c8d02
$ pip install .

References:
    https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
"""
from __future__ import absolute_import, division, print_function

import re
import argparse
import glob
import logging
import os
import pickle
import random
import json
import h5py
from easydict import EasyDict as edict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from utils.basic_utils import load_jsonl, flat_list_of_lists, save_jsonl

logger = logging.getLogger(__name__)


# only tested with roberta
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


def pad_sequences_1d(sequences, dtype=torch.long):
    """ Pad a single-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first dim has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
    """
    if isinstance(sequences[0], list):
        sequences = [torch.tensor(s, dtype=dtype) for s in sequences]
    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = torch.zeros(len(sequences), max(lengths)).float()
    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_collate(data):
    batch = edict()
    batch["text_ids"], batch["text_ids_mask"] = pad_sequences_1d([d["text_ids"] for d in data], dtype=torch.long)
    for k in data[0].keys():
        if k not in ["text_ids"]:
            batch[k] = [d[k] for d in data]
    # batch["unique_id"] = [d["unique_id"] for d in data]
    # batch["sen_lengths"] = [d["sen_lengths"] for d in data]
    return batch


class SingleSentenceDataset(Dataset):
    def __init__(self, tokenizer, input_datalist, block_size=512, add_extra_keys=False, debug=False):
        self.tokenizer = tokenizer
        self.max_length = block_size
        self.debug = debug
        self.debug_cnt = 100  # should be large than batch size
        self.add_extra_keys = add_extra_keys
        self.examples = self.read_examples(input_datalist, add_extra_keys)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def read_examples(self, input_datalist, add_extra_keys=False):
        """input_datalist, list(dict), each dict is,
        {"id": *str_id*,
        "text": raw consecutive text (e.g., a sentence or a paragraph) not processed in any way,}
        add_extra_keys: some additional keys that exist in the dict element of input_datalist,
            for example, "sen_lengths" for subtitles
        """
        examples = []
        for idx, line in tqdm(enumerate(input_datalist), desc="Loading data", total=len(input_datalist)):
            text_ids = self.single_segment_processor(line["text"], max_length=self.max_length)
            line_example = edict(id=line["id"],
                                 text=line["text"],
                                 text_ids=text_ids)
            if add_extra_keys:
                for ek in line.keys():
                    if ek not in line_example:
                        line_example[ek] = line[ek]
            examples.append(line_example)
            if self.debug and idx > self.debug_cnt:
                break
        return examples

    def single_segment_processor(self, single_segment_text, max_length):
        """
        single_segment_text: str, raw consecutive text (e.g., a sentence or a paragraph) not processed in any way
        Processing Steps:
        1) tokenize
        2) add special tokens
        # 3) pad to max length
        max_length: int, segment longer than max_length will be truncated
        """
        single_segment_ids = self.tokenizer.encode(single_segment_text,
                                                   add_special_tokens=True,
                                                   max_length=max_length)
        # sen_len = len(single_sentence_ids)
        # if sen_len < max_length:
        #     single_sentence_ids += [self.tokenizer.pad_token_id()] * (max_length - sen_len)
        #     pad_mask = [1] * sen_len + [0] * (max_length - sen_len)
        # else:
        #     pad_mask = [1] * max_length
        # return single_sentence_ids, pad_mask  # both are list(int), len == max_length
        return single_segment_ids


def load_preprocess_tvr_query(tvr_file_path):
    return [dict(id=e["desc_id"], text=e["desc"]) for e in load_jsonl(tvr_file_path)]


def extract_concat_sub_text(sub_info):
    """sub_info is list(dict), each dict is {"text": str, "start": float, "end": float}"""
    return " ".join([e["text"] for e in sub_info])


# def get_counter(tvr_subs):
#     sub_lengths = [len(tokenizer.tokenize(extract_concat_sub_text(e["sub"]))) for e in tqdm(tvr_subs)]


def assemble_chunked_sub_embeddings(chunked_h5_filepath, assemble_h5_filepath):
    """This can be seen as a inverse function of chunk_single_sub"""
    chunked_h5_f = h5py.File(chunked_h5_filepath, "r")
    assemble_h5_f = h5py.File(assemble_h5_filepath, "w")
    all_chunked_keys = chunked_h5_f.keys()
    vid_names = set([e.split("--")[0] for e in all_chunked_keys])
    embed_name_template = "{vid_name}--{chunk_idx}"
    for vname in tqdm(vid_names, desc="Assemble over video names"):
        # do not use `vname in e`, since it could match some other videos for TBBT
        vname_chunked_keys = [e for e in all_chunked_keys if e.startswith(vname)]
        num_chunks = int(len(vname_chunked_keys) / 2)
        assert len(vname_chunked_keys) % 2 == 0
        vname_embed_list = []
        vname_length_list = []
        for chunk_idx in range(num_chunks):
            chunk_name = embed_name_template.format(vid_name=vname, chunk_idx=chunk_idx)
            chunk_embed = chunked_h5_f[chunk_name][:]
            if len(chunk_embed.shape) == 1:
                chunk_embed = chunk_embed[None, :]
            vname_embed_list.append(chunk_embed)
            vname_length_list.append(chunked_h5_f[chunk_name + "-lengths"][:])
        vname_embed = np.concatenate(vname_embed_list, axis=0)  # (L, D)
        vname_lengths = np.concatenate(vname_length_list, axis=0)
        assert len(vname_embed) == sum(vname_lengths), \
            "len(vname_embed) {} sum(vname_lengths) {}".format(len(vname_embed), sum(vname_lengths))
        assemble_h5_f.create_dataset(vname, data=vname_embed, dtype=np.float32)
        assemble_h5_f.create_dataset(vname+"-lengths", data=vname_lengths, dtype=np.int16)  # the max values are less than 1000

    assemble_h5_f.close()
    chunked_h5_f.close()


def chunk_single_sub(tokenizer, single_sub_data, max_length=256, drop_edge=True):
    """
    tokenized sub length of 768 will cover 99.93% of all the subtitiles
    single_sub_data, dict {
        "vid_name": str,
        "sub": [{"text": " ( Chase :  ) That's all this is?", "start": 0.862, "end": 1.862},
                {'text': 'Yeah.', 'start': 1.988, 'end': 2.863},]
    }
    drop_edge: if True, drop small edges. This has be False for feature extraction
    """
    tokenized_text_list = [tokenizer.tokenize(e["text"]) for e in single_sub_data["sub"]]
    tokenized_lengths = [len(e) for e in tokenized_text_list]
    tokenized_lengths_cumsum = np.cumsum(tokenized_lengths)
    num_chunks = np.ceil(tokenized_lengths_cumsum[-1] / max_length).astype(np.int)
    if drop_edge and (tokenized_lengths_cumsum[-1] % max_length) / max_length < 0.1 and num_chunks > 1:
        # remove extra length if it is too short
        num_chunks -= 1
    start_cut_idx = 0
    pre_seq_len = 0
    chunked_sub_text = []  # list(sublist), each sublist is a list of tokens
    for chunk_idx in range(num_chunks):
        tokenized_lengths_cumsum -= pre_seq_len
        end_cut_idx = np.argmax(tokenized_lengths_cumsum >= max_length)
        if end_cut_idx == 0:
            end_cut_idx = len(tokenized_lengths)
        chunked_sub_text.append(dict(
            id="{}--{}".format(single_sub_data["vid_name"], chunk_idx),  # e.g., "s08e21_seg02_clip_10--0"
            text=flat_list_of_lists(tokenized_text_list[start_cut_idx: end_cut_idx]),
            sen_lengths=tokenized_lengths[start_cut_idx: end_cut_idx]
        ))
        assert sum(chunked_sub_text[-1]["sen_lengths"]) == len(chunked_sub_text[-1]["text"])
        assert len(chunked_sub_text[-1]["text"]) <= max_length, \
            "len {} st {} ed {}, max_length {}, st, chunk_idx {} \n lengths {} \ncumsum lengths {}\n text {}"\
            .format(len(chunked_sub_text[-1]["text"]), start_cut_idx, end_cut_idx,
                    max_length, chunk_idx, tokenized_lengths, tokenized_lengths_cumsum, " ".join(chunked_sub_text[-1]["text"]))
        pre_seq_len = sum(tokenized_lengths[start_cut_idx: end_cut_idx])
        start_cut_idx = end_cut_idx
    return chunked_sub_text


def load_preprocess_tvr_subtitles(tokenizer, sub_data_file, max_length,
                                  filter_file_path=None, drop_edge=True, debug=False):
    """
    filter_file_path: if provided, will be used to filter relevant subtitles
    max_chunks: int, split each subtitle into multiple chunks
    max_length: int,
    drop_edge, bool, must set to False when doing feature extraction, optionally set to True to save some time
    """
    sub_datalist = load_jsonl(sub_data_file)
    sub_datalist = sub_datalist[:100] if debug else sub_datalist
    if filter_file_path is not None:  # filter at finetuning, to use only subtitles in train set.
        assert len(filter_file_path) == 1, "please supply only one filter file path (--train_data_file)"
        filter_file_path = filter_file_path[0]
        keep_ids = list(set([e["vid_name"] for e in load_jsonl(filter_file_path)]))
        sub_datalist = [e for e in sub_datalist if e["vid_name"] in keep_ids]

    preprocessed_sub_datalist = flat_list_of_lists(
        [chunk_single_sub(tokenizer, sub_data, max_length=max_length, drop_edge=drop_edge)
         for sub_data in tqdm(sub_datalist, desc="Loading subtitles")])
    return preprocessed_sub_datalist


def load_and_cache_examples(tokenizer, data_path, max_length, sub_data_file=None, filter_file_path=None,
                            load_query=True, load_sub=False, extract_sub=False, add_extra_keys=False, debug=False):
    assert load_query or load_sub, "Must specify one of them"
    if extract_sub:
        assert not load_query, "Do not load query when extracting for subtitles"
    input_datalist = []
    if load_query:
        input_datalist = flat_list_of_lists([load_preprocess_tvr_query(e) for e in data_path])

    if load_sub:  # add sub data for training as well
        input_datalist += load_preprocess_tvr_subtitles(tokenizer,
                                                        sub_data_file=sub_data_file,
                                                        max_length=max_length-2,
                                                        filter_file_path=filter_file_path,
                                                        drop_edge=not extract_sub,
                                                        debug=debug)
    dataset = SingleSentenceDataset(tokenizer,
                                    input_datalist,
                                    block_size=max_length,
                                    add_extra_keys=add_extra_keys,
                                    debug=debug)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, inputs_pad_mask, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    inputs: torch.tensor (N, L)  # padded
    inputs_pad_mask: torch.tensor (N, L), `1` indicate valid tokens, `0` indicate <pad>
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()
    masked_indices[:, 0] = 0  # tokenizer.bos_token_id
    labels[~(masked_indices & inputs_pad_mask.bool())] = -1  # We only compute loss on masked non-pad tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=pad_collate)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps *
                (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = mask_tokens(batch.text_ids, batch.text_ids_mask, tokenizer, args)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            inputs_mask = batch.text_ids_mask.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels, attention_mask=inputs_mask)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # for distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if global_step > args.max_steps > 0:
                epoch_iterator.close()
                break

            if args.debug and step > 5:
                break

        if global_step > args.max_steps > 0:
            train_iterator.close()
            break

        if args.debug:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(tokenizer, args.eval_data_file, args.block_size, debug=args.debug)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=pad_collate)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for idx, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=len(eval_dataloader)):
        text_ids = batch.text_ids.to(args.device)
        text_ids_mask = batch.text_ids_mask.to(args.device)

        with torch.no_grad():
            outputs = model(text_ids, masked_lm_labels=text_ids, attention_mask=text_ids_mask)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

        if args.debug and idx > 5:
            break

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def get_batch_token_embeddings(layer_hidden_states, attention_mask, rm_special_tokens=False):
    """ remove padding and special tokens
    Args:
        layer_hidden_states: (N, L, D)
        attention_mask: (N, L) with 1 indicate valid bits, 0 pad bits
        rm_special_tokens: bool, whether to remove special tokens, this is different for different model_type
            1) a RoBERTa sequence has the following format: <s> X </s>
    return:
        list(np.ndarray), each ndarray is (L_sentence, D), where L_sentence <= L
    """
    valid_lengths = attention_mask.sum(1).long().tolist()  # (N, )
    layer_hidden_states = layer_hidden_states.cpu().numpy()
    embeddings = [e[1:vl-1] if rm_special_tokens else e[:vl]
                  for e, vl in zip(layer_hidden_states, valid_lengths)]
    return embeddings


def extract(args, model, tokenizer, prefix=""):
    """Many of the extraction args are inherited from evaluation"""
    extract_output_dir = args.output_dir

    extract_dataset = load_and_cache_examples(tokenizer, args.train_data_file, args.block_size,
                                              sub_data_file=args.sub_data_file,
                                              filter_file_path=None,
                                              load_query=not args.use_sub,
                                              load_sub=args.use_sub,
                                              extract_sub=args.use_sub,
                                              add_extra_keys=args.use_sub,
                                              debug=args.debug)

    if not os.path.exists(extract_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(extract_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    extract_sampler = SequentialSampler(extract_dataset) \
        if args.local_rank == -1 else DistributedSampler(extract_dataset)
    extract_dataloader = DataLoader(extract_dataset,
                                    sampler=extract_sampler,
                                    batch_size=args.eval_batch_size,
                                    collate_fn=pad_collate)

    # Eval!
    logger.info("***** Running extraction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(extract_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    if args.use_sub:
        extracted_file_name = args.extracted_file_name + ".chunked.h5"
    else:
        extracted_file_name = args.extracted_file_name
    output_extraction_file = os.path.join(extract_output_dir, extracted_file_name)
    assert not os.path.exists(output_extraction_file), "The file {} already exists".format(output_extraction_file)
    with h5py.File(output_extraction_file, "w") as h5_f:
        for idx, batch in tqdm(enumerate(extract_dataloader), desc="Extracting", total=len(extract_dataloader)):
            text_ids = batch.text_ids.to(args.device)
            text_ids_mask = batch.text_ids_mask.to(args.device)

            with torch.no_grad():
                outputs = model.roberta(text_ids, attention_mask=text_ids_mask)
                all_layer_hidden_states = outputs[2]
                extracted_hidden_states = get_batch_token_embeddings(
                    all_layer_hidden_states[-2], text_ids_mask, rm_special_tokens=True)
                if args.debug:
                    logger.info("outputs {}, all_layer_hiddens {}, -2 {}"
                                .format(len(outputs), len(all_layer_hidden_states), all_layer_hidden_states[-2].shape))
                    logger.info("last_to_second_layer_hidden_states {}"
                                .format([e.shape for e in extracted_hidden_states]))
            if args.debug and idx > 5:
                break

            for e_idx, (unique_id, text_feat) in enumerate(zip(batch.id, extracted_hidden_states)):
                h5_f.create_dataset(str(unique_id), data=text_feat, dtype=np.float32)

                if args.use_sub:
                    assert len(text_feat) == sum(batch.sen_lengths[e_idx]), \
                        "len(text_feat) {} == sum(batch.sen_lengths[e_idx]) {}"\
                        .format(len(text_feat), sum(batch.sen_lengths[e_idx]))
                    h5_f.create_dataset(str(unique_id)+"-lengths",
                                        data=np.array(batch.sen_lengths[e_idx]).astype(np.int16))

    if args.use_sub:
        logger.info("Start assemble chunked_sub_embeddings")
        assemble_h5_filepath = os.path.join(extract_output_dir, args.extracted_file_name)
        chunked_h5_filepath = os.path.join(extract_output_dir, args.extracted_file_name + ".chunked.h5")
        assert not os.path.exists(assemble_h5_filepath), "File {} already exists".format(assemble_h5_filepath)
        assemble_chunked_sub_embeddings(chunked_h5_filepath=chunked_h5_filepath,
                                        assemble_h5_filepath=assemble_h5_filepath)


def get_tokenized_text(args, tokenizer, prefix=""):
    """Many of the extraction args are inherited from evaluation"""
    extract_dataset = load_and_cache_examples(tokenizer, args.train_data_file, args.block_size,
                                              sub_data_file=args.sub_data_file,
                                              filter_file_path=None,
                                              load_query=True,
                                              debug=args.debug)

    # Eval!
    logger.info("***** Running Tokenization {} *****".format(prefix))
    logger.info("  Num examples = %d", len(extract_dataset))

    tokenized_desc_data = []  # each element is a dict {id: str, text: raw text string, tokens: tokenized text}
    for d in tqdm(extract_dataset):
        tokenized_d = dict(
            id=d["id"],
            text=d["text"],
            tokens=tokenizer.convert_ids_to_tokens(d["text_ids"], skip_special_tokens=True),
            tokenized_text=tokenizer.decode(d["text_ids"],
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False))
        tokenized_desc_data.append(tokenized_d)

    output_extraction_file = os.path.join(args.output_dir, args.extracted_file_name)
    save_jsonl(tokenized_desc_data, output_extraction_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_sub", action="store_true",
                        help="At finetuning: additionally add subtitles associated with the given train_data_file. "
                             "At extraction: extract embeddings for subtitles"
                             "Note using subtitles associated with val/test sets are not allowed!!!")
    parser.add_argument("--sub_data_file", type=str,
                        help="subtitle file, when --do_extract and --use_sub is set, extract from this file")

    parser.add_argument("--train_data_file", default=None, type=str, nargs="+",
                        help="The input training data file (a text file)"
                             "When --do_extract and not --use_sub, extract feature from this file(s).")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    # parser.add_argument("--extract_data_file", default=None, type=str,
    #                     help="An optional input extraction data file to extract the model embeddings, "
    #                          "the same format as the train_data_file/eval_data_file.")
    parser.add_argument("--extracted_file_name", default=None, type=str,
                        help=".h5 file name to save the extracted file")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 "
                             "(instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs "
                             "(take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_extract", action='store_true',
                        help="Whether to run extract")
    parser.add_argument("--do_tokenize", action="store_true",
                        help="Extract tokenized text. Typically used alone without loading models")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X updates steps., Will be override inside")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending "
                             "and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--debug', action='store_true', help="break all loops")
    args = parser.parse_args()

    if args.use_sub:
        assert os.path.exists(args.sub_data_file), "File {} does not exist".format(args.sub_data_file)

    if args.do_train:
        assert args.mlm, "Only work for MLM objective"
    assert args.model_type in ["roberta"], "Only work for roberta models"
    if args.do_extract:
        assert not args.do_train, "do_extract cannot be used with do_train"
        assert not args.do_eval, "do_extract cannot be used with do_eval"

    if args.do_train and args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. "
                         "Either supply a file to --eval_data_file or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. "
                         "Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    if args.do_extract:
        config.output_hidden_states = True  # output hidden states from all layers
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    if not args.do_tokenize:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()
        # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation/Extraction parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Barrier to make sure only the first process in distributed training process the dataset,
            # and the others will use the cache

        assert len(args.train_data_file) == 1, "Only use train split to finetune"
        train_dataset = load_and_cache_examples(tokenizer, args.train_data_file,
                                                max_length=args.block_size,
                                                sub_data_file=args.sub_data_file,
                                                filter_file_path=args.train_data_file,
                                                load_query=True,
                                                load_sub=args.use_sub,
                                                extract_sub=False,
                                                debug=args.debug)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer,
    # you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in
                               sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    # Extraction
    if args.do_extract and args.local_rank in [-1, 0]:
        logger.info("Extract features for {}"
                    .format(args.sub_data_file if args.use_sub else args.train_data_file))
        extract(args, model, tokenizer)

    # extract tokenized query
    if args.do_tokenize and args.local_rank in [-1, 0]:
        logger.info("Extract features for {}"
                    .format(args.sub_data_file if args.use_sub else args.train_data_file))
        get_tokenized_text(args, tokenizer)

    return results


if __name__ == "__main__":
    main()

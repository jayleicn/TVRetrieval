import numpy as np
import torch


def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_sequences_2d(sequences, dtype=torch.long):
    """ Pad a double-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor,
        only allow the first two dims has variable lengths
    Args:
        sequences: list(n-d tensor or list)
        dtype: torch.long for word indices / torch.float (float32) for other cases
    Returns:
    Examples:
        >>> test_data_list = [[[1, 3, 5], [3, 7, 4, 1]], [[98, 34, 11, 89, 90], [22], [34, 56]],]
        >>> pad_sequences_2d(test_data_list, dtype=torch.long)  # torch.Size([2, 3, 5])
        >>> test_data_3d = [torch.randn(2,2,4), torch.randn(4,3,4), torch.randn(1,5,4)]
        >>> pad_sequences_2d(test_data_3d, dtype=torch.float)  # torch.Size([2, 3, 5])
        >>> test_data_3d2 = [[torch.randn(2,4), ], [torch.randn(3,4), torch.randn(5,4)]]
        >>> pad_sequences_2d(test_data_3d2, dtype=torch.float)  # torch.Size([2, 3, 5])
    # TODO add support for numpy array
    """
    bsz = len(sequences)
    para_lengths = [len(seq) for seq in sequences]
    max_para_len = max(para_lengths)
    sen_lengths = [[len(word_seq) for word_seq in seq] for seq in sequences]
    max_sen_len = max([max(e) for e in sen_lengths])

    if isinstance(sequences[0], torch.Tensor):
        extra_dims = sequences[0].shape[2:]
    elif isinstance(sequences[0][0], torch.Tensor):
        extra_dims = sequences[0][0].shape[1:]
    else:
        sequences = [[torch.Tensor(word_seq, dtype=dtype) for word_seq in seq] for seq in sequences]
        extra_dims = ()

    padded_seqs = torch.zeros((bsz, max_para_len, max_sen_len) + extra_dims, dtype=dtype)
    mask = torch.zeros(bsz, max_para_len, max_sen_len).float()

    for b_i in range(bsz):
        for sen_i, sen_l in enumerate(sen_lengths[b_i]):
            padded_seqs[b_i, sen_i, :sen_l] = sequences[b_i][sen_i]
            mask[b_i, sen_i, :sen_l] = 1
    return padded_seqs, mask  # , sen_lengths


def find_max_triples(st_prob, ed_prob, top_n=5, prob_thd=None, tensor_type="torch"):
    """ Find a list of (k1, k2) where k1 < k2 with the maximum values of st_prob[k1] * ed_prob[k2]
    Args:
        st_prob (torch.Tensor or np.ndarray): (N, L) batched start_idx probabilities
        ed_prob (torch.Tensor  or np.ndarray): (N, L) batched end_idx probabilities
        top_n (int): return topN pairs with highest values
        prob_thd (float):
        tensor_type: str, np or torch
    Returns:
        batched_sorted_triple: N * [(st_idx, ed_idx, confidence), ...]
    """
    if tensor_type == "torch":
        st_prob, ed_prob = st_prob.data.numpy(), ed_prob.data.numpy()
    product = np.einsum("bm,bn->bmn", st_prob, ed_prob)
    # (N, L, L) the lower part becomes zeros, start_idx < ed_idx
    upper_product = np.triu(product, k=1)
    return find_max_triples_from_upper_triangle_product(upper_product, top_n=top_n, prob_thd=prob_thd)


def find_max_triples_from_upper_triangle_product(upper_product, top_n=5, prob_thd=None):
    """ Find a list of (k1, k2) where k1 < k2 with the maximum values of p1[k1] * p2[k2]
    Args:
        upper_product (torch.Tensor or np.ndarray): (N, L, L), the lower part becomes zeros, end_idx > start_idx
        top_n (int): return topN pairs with highest values
        prob_thd (float or None):
    Returns:
        batched_sorted_triple: N * [(st_idx, ed_idx, confidence), ...]
    """
    batched_sorted_triple = []
    for idx, e in enumerate(upper_product):
        sorted_triple = top_n_array_2d(e, top_n=top_n)
        if prob_thd is not None:
            sorted_triple = sorted_triple[sorted_triple[2] >= prob_thd]
        batched_sorted_triple.append(sorted_triple)
    return batched_sorted_triple


def top_n_array_2d(array_2d, top_n):
    """ Get topN indices and values of a 2d array, return a tuple of indices and their values,
    ranked by the value
    """
    row_indices, column_indices = np.unravel_index(np.argsort(array_2d, axis=None), array_2d.shape)
    row_indices = row_indices[::-1][:top_n]
    column_indices = column_indices[::-1][:top_n]
    sorted_values = array_2d[row_indices, column_indices]
    return np.stack([row_indices, column_indices, sorted_values], axis=1)  # (N, 3)

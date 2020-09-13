import math
import numpy
import torch
import time
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union


def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns segmented spans in the target with respect to the provided span indices.
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.
    # Returns
    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(target)).view(
        1, 1, -1
    )
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    raw_span_indices = span_starts + max_span_range_indices
    # We also don't want to include span indices which greater than the sequence_length,
    # which happens because some spans near the end of the sequence
    # have a start index + max_batch_span_width > sequence_length, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices < target.size(1)) & (0 <= raw_span_indices)
    span_indices = raw_span_indices * span_mask

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def batched_index_select(
        target: torch.Tensor,
        indices: torch.LongTensor,
        flattened_indices: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns selected values in the target with respect to the provided indices, which
    have size `(batch_size, d_1, ..., d_n, embedding_size)`. This can use the optionally
    precomputed `flattened_indices` with size `(batch_size * d_1 * ... * d_n)` if given.
    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    [CoreferenceResolver](../models/coreference_resolution/coref.md). Model to select
    contextual word representations corresponding to the start and end indices of mentions. The key
    reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A tensor of shape (batch_size, ...), where each element is an index into the
        `sequence_length` dimension of the `target` tensor.
    flattened_indices : `Optional[torch.Tensor]`, optional (default = `None`)
        An optional tensor representing the result of calling `flatten_and_batch_shift_indices`
        on `indices`. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.
    # Returns
    selected_targets : `torch.Tensor`
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for [`batched_index_select`](./util.md#batched_index_select).
    The given `indices` of size `(batch_size, d_1, ..., d_n)` indexes into dimension 2 of a
    target tensor, which has size `(batch_size, sequence_length, embedding_size)`. This
    function returns a vector that correctly indexes into the flattened target. The sequence
    length of the target must be provided to compute the appropriate offsets.
    ```python
        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]
    ```
    # Parameters
    indices : `torch.LongTensor`, required.
    sequence_length : `int`, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.
    # Returns
    offset_indices : `torch.LongTensor`
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise Exception(
            f"All elements in indices should be in range (0, {sequence_length - 1})"
        )
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def pool_span_representation(input_hidden, offsets):
    '''
    :param input_hidden: `torch.FloatTensor`
            Shape: [batch_size, num_wordpieces, embedding_size].
    :param offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
    :param orig_tokens_mask:`torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
    :return orig_embeddings: `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
    '''
    span_embeddings, span_mask = batched_span_select(input_hidden.contiguous(), offsets)
    span_mask = span_mask.unsqueeze(-1)
    span_embeddings *= span_mask  # zero out paddings

    span_embeddings_sum = span_embeddings.sum(2)
    span_embeddings_len = span_mask.sum(2)
    # Shape: (batch_size, num_orig_tokens, embedding_size)
    orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

    # All the places where the span length is zero, write in zeros.
    orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

    return orig_embeddings


def sort_by_seq_len(data, pad):
    # 对text 和 tag 按seq_len排序，便于使用pack_padded_sequence
    # 对序列长度进行排序(降序), sorted_seq_lengths = [5, 3, 2]
    # indices 为 [1, 0, 2], indices 的值可以这么用语言表述
    # 原来 batch 中在 0 位置的值, 现在在位置 1 上.
    # 原来 batch 中在 1 位置的值, 现在在位置 0 上.
    # 原来 batch 中在 2 位置的值, 现在在位置 2 上.
    seq_lengths = get_seq_len_batch(data, pad)
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
    # 如果我们想要将计算的结果恢复排序前的顺序的话,
    # 只需要对 indices 再次排序(升序),会得到 [0, 1, 2],
    # desorted_indices 的结果就是 [1, 0, 2]
    # 使用 desorted_indices 对计算结果进行索引就可以了.
    _, desorted_indices = torch.sort(indices, descending=False)
    return indices, desorted_indices


def get_seq_len_batch(data, pad):
    '''
    计算batch中每个句子的真实长度
    :param data:pad后的batch[batch_sz,seq_len]
    :return:[batch_sz]
    '''
    seq_lengths = data.gt(pad).sum(1)
    return seq_lengths


def time_since(start_time):
    return time.time() - start_time


if __name__ == '__main__':
    h = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]).unsqueeze(-1)
    print(h)
    offsets = torch.LongTensor([[[0, 0], [1, 1], [2, 2]], [[0, 1], [2, 2], [-1, -1]]])
    print(offsets)
    data = pool_span_representation(h, offsets)
    print(data)

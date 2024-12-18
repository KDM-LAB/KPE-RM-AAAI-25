import os
import sys
import json
import torch
import logging
import traceback
from tqdm import tqdm
from . import loader_utils
from ..constant import BOS_WORD, EOS_WORD

logger = logging.getLogger()


# -------------------------------------------------------------------------------------------
# preprocess label
# ------------------------------------------------------------------------------------------
def convert_to_label(filter_positions, tot_mention_list, differ_phrase_num):
    """First check keyphrase mentions index is same ;
    Then set keyprhase ngrams = +1  and other phrase candidates = -1 .
    """
    ngram_label = [-1 for _ in range(differ_phrase_num)]
    chunk_label_list = [
        [0] * len(tot_mention_list[i]) for i in range(len(tot_mention_list))
    ]

    for i, positions in enumerate(filter_positions):
        for s, e in positions:
            chunk_label_list[e - s][s] = 1
            key_index = tot_mention_list[e - s][s]
            ngram_label[key_index] = 1

    # flat chunk label
    chunk_label = [_chunk for chunks in chunk_label_list for _chunk in chunks]

    # keep have more than one positive and one negtive
    if (
        (1 in ngram_label)
        and (-1 in ngram_label)
        and (1 in chunk_label)
        and (0 in chunk_label)
    ):
        return ngram_label, chunk_label
    else:
        return None, None


# -------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


def get_ngram_features(doc_words, max_gram, stem_flag=False):

    phrase2index = {}  # use to shuffle same phrases
    tot_phrase_list = []  # use to final evaluation
    tot_mention_list = []  # use to train pooling the same

    gram_num = 0
    for n in range(1, max_gram + 1):
        valid_length = len(doc_words) - n + 1

        if valid_length < 1:
            break

        _ngram_list = []
        _mention_list = []
        for i in range(valid_length):

            gram_num += 1

            n_gram = " ".join(doc_words[i : i + n]).lower()

            if stem_flag:
                index = loader_utils.whether_stem_existing(
                    n_gram, phrase2index, tot_phrase_list
                )
            else:
                index = loader_utils.whether_existing(
                    n_gram, phrase2index, tot_phrase_list
                )

            _mention_list.append(index)
            _ngram_list.append(n_gram)

        tot_mention_list.append(_mention_list)

    assert len(tot_phrase_list) > 0

    assert (len(tot_phrase_list) - 1) == max(tot_mention_list[-1])
    assert sum([len(_mention_list) for _mention_list in tot_mention_list]) == gram_num
    return {"tot_phrase_list": tot_phrase_list, "tot_mention_list": tot_mention_list}


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def get_ngram_chunk_features(doc_words, max_gram, keyphrases, stem_flag=False):

    keyphrases_list = [" ".join(kp).lower() for kp in keyphrases]

    chunk_label = []

    phrase2index = {}  # use to shuffle same phrases
    tot_phrase_list = []  # use to final evaluation
    tot_mention_list = []  # use to train pooling the same

    gram_num = 0
    for n in range(1, max_gram + 1):
        valid_length = len(doc_words) - n + 1

        if valid_length < 1:
            break

        _ngram_list = []
        _mention_list = []
        for i in range(valid_length):

            gram_num += 1

            n_gram = " ".join(doc_words[i : i + n]).lower()

            if stem_flag:
                index = loader_utils.whether_stem_existing(
                    n_gram, phrase2index, tot_phrase_list
                )
            else:
                index = loader_utils.whether_existing(
                    n_gram, phrase2index, tot_phrase_list
                )

            # -----------------------------------------------------------------
            # chunk label
            if n_gram in keyphrases_list:
                chunk_label.append(1)
            else:
                chunk_label.append(0)
            # -----------------------------------------------------------------

            _mention_list.append(index)
            _ngram_list.append(n_gram)

        tot_mention_list.append(_mention_list)

    assert len(tot_phrase_list) > 0

    assert (len(tot_phrase_list) - 1) == max(tot_mention_list[-1])
    assert (
        sum([len(_mention_list) for _mention_list in tot_mention_list])
        == gram_num
        == len(chunk_label)
    )
    return {
        "tot_phrase_list": tot_phrase_list,
        "tot_mention_list": tot_mention_list,
        "chunk_label": chunk_label,
    }


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


def get_ngram_info_label(
    doc_words, max_phrase_words, stem_flag, keyphrases=None, start_end_pos=None
):

    returns = {"overlen_flag": False, "ngram_label": None, "chunk_label": None}
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    feature = get_ngram_features(
        doc_words=doc_words, max_gram=max_phrase_words, stem_flag=stem_flag
    )
    returns["tot_phrase_list"] = feature["tot_phrase_list"]
    returns["tot_mention_list"] = feature["tot_mention_list"]

    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    if start_end_pos is not None:
        filter_positions = loader_utils.limit_scope_length(
            start_end_pos, len(doc_words), max_phrase_words
        )

        # check over_length
        if len(filter_positions) != len(start_end_pos):
            returns["overlen_flag"] = True

        if len(filter_positions) > 0:
            returns["ngram_label"], returns["chunk_label"] = convert_to_label(
                **{
                    "filter_positions": filter_positions,
                    "tot_mention_list": feature["tot_mention_list"],
                    "differ_phrase_num": len(feature["tot_phrase_list"]),
                }
            )
        else:
            returns["ngram_label"] = None
            returns["chunk_label"] = None

    return returns


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
def bert2joint_preprocessor(
    examples,
    tokenizer,
    max_token,
    pretrain_model,
    mode,
    max_phrase_words,
    stem_flag=False,
):
    logger.info(
        "start preparing (%s) features for bert2joint (%s) ..." % (mode, pretrain_model)
    )

    overlen_num = 0
    new_examples = []
    # for idx, ex in enumerate(tqdm(examples)):
    for idx, ex in enumerate(examples):

        # tokenize
        tokenize_output = loader_utils.tokenize_for_bert(
            doc_words=ex["doc_words"], tokenizer=tokenizer
        )

        if len(tokenize_output["tokens"]) < max_token:
            max_word = max_token
        else:
            max_word = tokenize_output["tok_to_orig_index"][max_token - 1] + 1

        new_ex = {}
        new_ex["url"] = ex["url"]
        new_ex["tokens"] = tokenize_output["tokens"][:max_token]
        new_ex["valid_mask"] = tokenize_output["valid_mask"][:max_token]
        new_ex["doc_words"] = ex["doc_words"][:max_word]
        assert len(new_ex["tokens"]) == len(new_ex["valid_mask"])
        assert sum(new_ex["valid_mask"]) == len(new_ex["doc_words"])

        # ---------------------------------------------------------------------------
        parameter = {
            "doc_words": new_ex["doc_words"],
            "max_phrase_words": max_phrase_words,
            "stem_flag": stem_flag,
        }
        # ---------------------------------------------------------------------------
        if mode == "train":
            parameter["keyphrases"] = ex["keyphrases"]
            parameter["start_end_pos"] = ex["start_end_pos"]
        # ---------------------------------------------------------------------------
        # obtain gram info and label
        info_or_label = get_ngram_info_label(**parameter)

        new_ex["phrase_list"] = info_or_label["tot_phrase_list"]
        new_ex["mention_lists"] = info_or_label["tot_mention_list"]

        if info_or_label["overlen_flag"]:
            overlen_num += 1
        # ---------------------------------------------------------------------------
        if mode == "train":
            if not info_or_label["ngram_label"]:
                continue
            new_ex["keyphrases"] = ex["keyphrases"]
            new_ex["ngram_label"] = info_or_label["ngram_label"]
            new_ex["chunk_label"] = info_or_label["chunk_label"]
        # ---------------------------------------------------------------------------
        new_examples.append(new_ex)

    logger.info(
        "Delete Overlen Keyphrase (length > 5): %d (overlen / total = %.2f"
        % (overlen_num, float(overlen_num / len(examples) * 100))
        + "%)"
    )
    return new_examples


# ********************************************************************************************************
# ********************************************************************************************************
def bert2joint_converter(index, ex, tokenizer, mode, max_phrase_words):
    """ convert each batch data to tensor ; add [CLS] [SEP] tokens ;"""

    src_tokens = [BOS_WORD] + ex["tokens"] + [EOS_WORD]
    valid_ids = [0] + ex["valid_mask"] + [0]

    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
    valid_mask = torch.LongTensor(valid_ids)

    mention_lists = ex["mention_lists"]
    orig_doc_len = sum(valid_ids)

    if mode == "train":
        label = torch.LongTensor(ex["ngram_label"])
        chunk_label = torch.LongTensor(ex["chunk_label"])
        return (
            index,
            src_tensor,
            valid_mask,
            mention_lists,
            orig_doc_len,
            max_phrase_words,
            label,
            chunk_label,
        )

    else:
        tot_phrase_len = len(ex["phrase_list"])
        return (
            index,
            src_tensor,
            valid_mask,
            mention_lists,
            orig_doc_len,
            max_phrase_words,
            tot_phrase_len,
        )


def batchify_bert2joint_features_for_train(batch):
    """ train dataloader & eval dataloader ."""

    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    mention_mask = [ex[3] for ex in batch]
    doc_word_lens = [ex[4] for ex in batch]
    max_phrase_words = [ex[5] for ex in batch][0]

    # label
    label_list = [ex[6] for ex in batch]  # different ngrams numbers
    chunk_list = [ex[7] for ex in batch]  # whether is a chunk phrase

    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

    # ---------------------------------------------------------------
    # [1] [2] src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
    # segment_ids = torch.LongTensor(len(docs), doc_max_length).zero_()

    for i, d in enumerate(docs):
        input_ids[i, : d.size(0)].copy_(d)
        input_mask[i, : d.size(0)].fill_(1)

    # ---------------------------------------------------------------
    # [3] valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, : v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # [4] active mention mask : for n-gram (original)

    max_ngram_length = sum([max_word_len - n for n in range(max_phrase_words)])
    chunk_mask = torch.LongTensor(len(docs), max_ngram_length).fill_(-1)

    for batch_i, word_len in enumerate(doc_word_lens):
        pad_len = max_word_len - word_len

        batch_mask = []
        for n in range(max_phrase_words):
            ngram_len = word_len - n
            if ngram_len > 0:
                assert len(mention_mask[batch_i][n]) == ngram_len
                gram_list = mention_mask[batch_i][n] + [
                    -1 for _ in range(pad_len)
                ]  # -1 for padding
            else:
                gram_list = [-1 for _ in range(max_word_len - n)]
            batch_mask.extend(gram_list)
        chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))

    # ---------------------------------------------------------------
    # [4] active mask : for n-gram
    max_diff_gram_num = 1 + max(
        [max(_mention_mask[-1]) for _mention_mask in mention_mask]
    )
    active_mask = torch.BoolTensor(
        len(docs), max_diff_gram_num, max_ngram_length
    ).fill_(
        1
    )  # Pytorch Version = 1.1.3
    #     active_mask = torch.ByteTensor(len(docs), max_diff_gram_num, max_ngram_length).fill_(1) # Pytorch Version = 1.1.0
    for gram_ids in range(max_diff_gram_num):
        tmp = torch.where(
            chunk_mask == gram_ids,
            torch.LongTensor(len(docs), max_ngram_length).fill_(0),
            torch.LongTensor(len(docs), max_ngram_length).fill_(1),
        )  # shape = (batch_size, max_ngram_length) # 1 for pad
        for batch_id in range(len(docs)):
            active_mask[batch_id][gram_ids].copy_(tmp[batch_id])

    # -------------------------------------------------------------------
    # [5] label : for n-gram
    max_diff_grams_num = max([label.size(0) for label in label_list])
    ngram_label = torch.LongTensor(len(label_list), max_diff_grams_num).zero_()
    for batch_i, label in enumerate(label_list):
        ngram_label[batch_i, : label.size(0)].copy_(label)

    # -------------------------------------------------------------------
    # [6] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)

    # -------------------------------------------------------------------
    # [7] Chunk Lable :
    max_chunks_num = max([chunks.size(0) for chunks in chunk_list])
    chunk_label = torch.LongTensor(len(chunk_list), max_chunks_num).fill_(-1)
    for batch_i, chunks in enumerate(chunk_list):
        chunk_label[batch_i, : chunks.size(0)].copy_(chunks)

    return (
        input_ids,
        input_mask,
        valid_ids,
        active_mask,
        valid_output,
        ngram_label,
        chunk_label,
        chunk_mask,
        ids,
    )


def batchify_bert2joint_features_for_test(batch):
    """ test dataloader for Dev & Public_Valid."""

    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    mention_mask = [ex[3] for ex in batch]
    doc_word_lens = [ex[4] for ex in batch]
    max_phrase_words = [ex[5] for ex in batch][0]

    phrase_list_lens = [ex[6] for ex in batch]

    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens])  # word-level

    # ---------------------------------------------------------------
    # [1] [2] src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()
    # segment_ids = torch.LongTensor(len(docs), doc_max_length).zero_()

    for i, d in enumerate(docs):
        input_ids[i, : d.size(0)].copy_(d)
        input_mask[i, : d.size(0)].fill_(1)

    # ---------------------------------------------------------------
    # [3] valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, : v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # [4] active mention mask : for n-gram (original)

    max_ngram_length = sum([max_word_len - n for n in range(max_phrase_words)])
    chunk_mask = torch.LongTensor(len(docs), max_ngram_length).fill_(-1)

    for batch_i, word_len in enumerate(doc_word_lens):
        pad_len = max_word_len - word_len

        batch_mask = []
        for n in range(max_phrase_words):
            ngram_len = word_len - n
            if ngram_len > 0:
                assert len(mention_mask[batch_i][n]) == ngram_len
                gram_list = mention_mask[batch_i][n] + [
                    -1 for _ in range(pad_len)
                ]  # -1 for padding
            else:
                gram_list = [-1 for _ in range(max_word_len - n)]
            batch_mask.extend(gram_list)
        chunk_mask[batch_i].copy_(torch.LongTensor(batch_mask))

    # ---------------------------------------------------------------
    # [4] active mask : for n-gram
    max_diff_gram_num = 1 + max(
        [max(_mention_mask[-1]) for _mention_mask in mention_mask]
    )
    active_mask = torch.BoolTensor(
        len(docs), max_diff_gram_num, max_ngram_length
    ).fill_(1)
    #     active_mask = torch.ByteTensor(len(docs), max_diff_gram_num, max_ngram_length).fill_(1) # Pytorch Version = 1.1.0
    for gram_ids in range(max_diff_gram_num):
        tmp = torch.where(
            chunk_mask == gram_ids,
            torch.LongTensor(len(docs), max_ngram_length).fill_(0),
            torch.LongTensor(len(docs), max_ngram_length).fill_(1),
        )  # shape = (batch_size, max_ngram_length) # 1 for pad
        for batch_id in range(len(docs)):
            active_mask[batch_id][gram_ids].copy_(tmp[batch_id])

    # -------------------------------------------------------------------
    # [5] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim)
    return (
        input_ids,
        input_mask,
        valid_ids,
        active_mask,
        valid_output,
        phrase_list_lens,
        ids,
    )

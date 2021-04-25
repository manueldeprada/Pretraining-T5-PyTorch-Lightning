import json
import math
import os
import random
from itertools import islice
from multiprocessing import Process
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib
import typer
from tqdm import tqdm
from transformers import T5Tokenizer

DATA_FILE = "data/cord19-standard.txt"
DATASET_CACHE_PATH = Path("dataset_cache/")
DATASET_CACHE_PATH.mkdir(exist_ok=True, parents=True)

dot_token: int
dot_token_1: int
mask_tokens: list


def process_sequence(sequence, processed_sequences, tokenizer):
    """This function takes a sequence of input ids (a tokenized sentence) and slices it if it is too long.
        It also discards the sequence if it cannot be sliced or has too much noise."""
    number_unknowns = sequence.count(tokenizer.unk_token_id)
    if number_unknowns > 0.05 * len(sequence) or len(sequence) < 5:
        return  # discard sentence if too much unknown tokens or too short sentence
    if len(sequence) > tokenizer.model_max_length:
        try:
            slice_sequence(sequence, processed_sequences, tokenizer.model_max_length)
        except:  # if we cannot slice by . we give up and discard the sentence
            return
    else:
        processed_sequences.append(sequence)


def slice_sequence(sequence, processed_sequences, max_length):
    """This is a recursive function that takes a sequence and slices it until it fits the model searching for dots."""
    try:
        last_dot = list_last_index(sequence[:max_length - 1], dot_token) + 1
    except:  # there is no ., we try with ).
        last_dot = list_last_index(sequence[:max_length - 1], dot_token_1) + 1  # dot_token_1 corresponds to ). token
    processed_sequences.append(sequence[:last_dot])
    sequence = sequence[last_dot:]
    if len(sequence) > max_length:
        slice_sequence(sequence, processed_sequences, max_length)
    else:
        processed_sequences.append(sequence)


def list_last_index(li, x):
    """Efficiently searches backwards for the first occurrence of x in a list."""
    for i in reversed(range(len(li))):
        if li[i] == x:
            return i
    raise ValueError("{} is not in list".format(x))


def generate_target_ids(input_ids, mask_prob):
    """This function takes a list of sentences and generates the pair (input_ids, target_ids) for pretraining the
    model. It implements in a simple way the final T5 denoising objective, as per HuggingFace documentation.

    :param mask_prob: Probability of masking a token.
    :param input_ids: A list of sublists, where the sublists are sequences of input ids (tokenized sentences). This
        mutable sublists are modified within this function, masking the tokens that the model has to denoise for
        pretraining.
    :return: The correspondent target sequences of ids for each input sentence, with the unmasked tokens.
    """
    target_ids = []
    for _input_sent_embed in tqdm(input_ids):  # let's calculate masks for denoising pretraining
        _target_sent_embed = []
        masked_indexes = sorted(random.sample(range(0, len(_input_sent_embed)),  # sample a word index in sentence
                                              min(int(mask_prob * len(_input_sent_embed)),  # number of tokens masked
                                                  len(mask_tokens) - 1)))  # but never more than special tokens available
        mask = [(i in masked_indexes)  # this is True or False
                for i in range(len(_input_sent_embed))]
        i = 0
        end = len(_input_sent_embed)
        masked_spans_counter = 0
        while i < end:
            if mask[i]:
                current_words_masked = [_input_sent_embed[i]]
                _input_sent_embed[i] = mask_tokens[masked_spans_counter]
                masked_spans_counter += 1
                while i + 1 < end and mask[i + 1]:
                    current_words_masked.append(_input_sent_embed[i + 1])
                    del _input_sent_embed[i + 1]
                    del mask[i + 1]
                    end -= 1
                _target_sent_embed.extend(current_words_masked)
            else:
                if len(_target_sent_embed) == 0 or _target_sent_embed[-1] != mask_tokens[masked_spans_counter]:
                    _target_sent_embed.append(mask_tokens[masked_spans_counter])
            i += 1
        target_ids.append(_target_sent_embed)
    return target_ids


def write_disk(input_ids, target_ids, file_counter):
    print("New thread: writing file: " + str(DATASET_CACHE_PATH / (Path("dataset_" + str(file_counter)).stem + ".jbl")))
    joblib.dump([input_ids, target_ids],  # performance bottleneck 2 here. Now in separate process
                DATASET_CACHE_PATH / (Path("dataset_" + str(file_counter)).stem + ".jbl"))
    # open(CACHE_PATH / (Path("test").stem + ".jbl.example"), "w").write(str([input_ids, target_ids]))
    # print("\rFile written: " + str(CACHE_PATH / (Path("dataset_" + str(file_counter)).stem + ".jbl")))


def main(tokenizer_name: str = typer.Argument("t5-base", help="T5 tokenizer used for token ids."),
         valid_size: float = typer.Argument(0.2, help="Validation set size."),
         dumps_size: int = typer.Argument(100, help="Size in MB for the dataset raw files."),
         mask_probability: float = typer.Argument(0.15, help="Probability of masking a token in a sentence.")):
    """This script preprocesses and tokenizes a standardized pretraining text Dataset (a file with a sentence in each
    line) into a set of tokenized files for training and validating the text2text model."""
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    global dot_token, dot_token_1, mask_tokens
    dot_token = tokenizer.convert_tokens_to_ids(["."])[0]
    dot_token_1 = tokenizer.convert_tokens_to_ids([")."])[0]
    mask_tokens = tokenizer.additional_special_tokens_ids
    meta = {}
    words_per_dump = 300_000 * dumps_size  # approx. 300_000 words per mb of dump file.
    with open(DATA_FILE, 'r') as in_file:
        number_lines = len([0 for _ in in_file])
        in_file.seek(0)  # after reading number of lines, restart file pointer
        n = 100000  # size of batches of sentences from input file. ~=100mb chunks
        batch_counter, file_counter, words_counter = 1, 1, 0
        input_ids, target_ids = [], []
        for sentence_batch in iter(lambda: tuple(islice(in_file, n)), ()):  # tuples of islices size n until tuple ()
            print(f"Processing batch {batch_counter} of {math.ceil(number_lines / n)}.")
            inputs_batch = tokenizer.batch_encode_plus(sentence_batch, return_attention_mask=False, verbose=False)[
                "input_ids"]  # performance bottleneck 1 here
            processed_batch = []
            for sequence in inputs_batch:  # input batches of 100k elements may have different sizes once processed
                process_sequence(sequence, processed_batch, tokenizer)
            del inputs_batch
            input_ids.extend(processed_batch)
            target_ids.extend(generate_target_ids(processed_batch, mask_probability))
            for x in processed_batch: words_counter += len(x)
            del processed_batch
            if words_counter > words_per_dump:  # 30M words ~= 100MB dump file size
                dump_size = int(len(input_ids) * words_per_dump / words_counter)
                meta[f"dataset_{file_counter}.jbl"] = dump_size
                Process(target=write_disk, args=(input_ids[:dump_size], target_ids[:dump_size], file_counter)).start()
                input_ids, target_ids = input_ids[dump_size:], target_ids[dump_size:]
                file_counter += 1
                words_counter -= words_per_dump
            batch_counter += 1
        Process(target=write_disk, args=(input_ids, target_ids, file_counter)).start()  # write last dump to disk
        meta[f"dataset_{str(file_counter)}.jbl"] = len(input_ids)
    print("Dataset tokenized. Partitioning...")
    total_size = sum(meta.values())
    train, valid = [], []
    count, train_size = 0, 0
    for file, size in meta.items():
        count += size
        if count < (1 - valid_size) * total_size:
            train_size += size
            train.append(file)
        else:
            valid.append(file)
    meta["train_size"], meta["valid_size"] = train_size, total_size - train_size
    meta["train"], meta["valid"] = train, valid
    with open(DATASET_CACHE_PATH / Path("dataset_meta.json"), 'w') as json_file:
        json.dump(meta, json_file, indent=2)
    print("Dataset ready. Meta file written to " + str(DATASET_CACHE_PATH / Path("dataset_meta.json")))


if __name__ == "__main__":
    typer.run(main)

import torch.utils.data as Data
import torch
import numpy as np


class MLMDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos, attention_mask):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.attention_mask = attention_mask
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.attention_mask[idx]

class GANDataSet(Data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

class iterator():
    def __init__(self, dataset) -> None:
        self.iterators = None
        self.dataset = dataset
    def get_iter(self):
        self.iterators = iter(self.dataset)
    def get_batch(self):
        if self.iterators is None:
            self.get_iter()
        try:
            batch = next(self.iterators)
        except StopIteration:
            self.get_iter()
            batch = next(self.iterators)
        return batch
 

 

def generate_attention_mask(input_ids, pad_token_id):
    return (input_ids != pad_token_id).long()

def pad_sequences(sequences, max_length, pad_token_id):
    """
    Pad the list of sequences (numerical token ids) to the same length.
    Sequence that are shorter than the specified ``max_len`` will be appended
    with the specified ``pad_token_id``. Those that are longer will be truncated.

    Parameters
    ----------
    sequences : list[int]
        List of numerical token ids.

    max_length : int
         Maximum length that all sequences will be truncated/padded to.

    pad_token_id : int
        Padding token index.

    Returns
    -------
    padded_sequences : 1d ndarray
    """
    num_samples = len(sequences)
    padded_sequences = np.full((num_samples, max_length), pad_token_id)
    for i, sequence in enumerate(sequences):
        sequence = np.array(sequence)[:max_length]
        padded_sequences[i, :len(sequence)] = sequence

    return padded_sequences

class Seq2SeqDataCollator:
    def __init__(
        self,
        max_length: int,
        pad_token_id: int,
        pad_label_token_id: int = -100
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_label_token_id = pad_label_token_id
        
    def __call__(self, batch):
        source_batch = []
        source_len = []
        target_batch = []
        target_len = []
        for example in batch:
            source = example['input_ids']
            source_len.append(int(len(source)))
            source_batch.append(source)

            target = example['labels']
            target_len.append(len(target))
            target_batch.append(target)

        source_padded = self.process_encoded_text(source_batch, source_len, self.pad_token_id)
        target_padded = self.process_encoded_text(target_batch, target_len, self.pad_label_token_id)
        attention_mask = generate_attention_mask(source_padded, self.pad_token_id)
        return {
            'input_ids': source_padded,
            'labels': target_padded,
            'attention_mask': attention_mask,
        }

    def process_encoded_text(self, sequences, sequences_len, pad_token_id):
        sequences_max_len = np.max(sequences_len)
        max_length = min(sequences_max_len, self.max_length)
        padded_sequences = pad_sequences(sequences, max_length, pad_token_id)
        return torch.LongTensor(padded_sequences)




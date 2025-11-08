
"""Code is adapted from MEZO: https://github.com/princeton-nlp/MeZO"""

import os
import torch
import pandas as pd
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers import RobertaTokenizer

MAX_LEN = 256
PRIVATE_SAMPLE_PER_CLASS = 512
PUBLIC_SAMPLE_PER_CLASS = 100
first_sent_limit = 240
other_sent_limit = 256

class MnliProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, k):
        """See base class."""
        return self._create_fewshot_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", k)

    def get_dev_examples(self, data_dir, size):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched", size)

    def get_test_examples(self, data_dir, size):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched", size)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, size):
        """Creates examples of a given size."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0: continue
            if i == size+1: break
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    def _create_fewshot_examples(self, lines, set_type, k):
        """Creates k examples per class."""
        examples = []
        for target_label in ["contradiction", "entailment", "neutral"]:
            count = 0
            for (i, line) in enumerate(lines):
                if i == 0: continue
                label = line[-1]
                if label != target_label: continue
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[8]
                text_b = line[9]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                count += 1
                if count == k: break
        return examples

class SnliProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, k):
        """See base class."""
        return self._create_fewshot_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", k)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_fewshot_examples(self, lines, set_type, k):
        """Creates k examples per class."""
        examples = []
        for target_label in ["contradiction", "entailment", "neutral"]:
            count = 0
            for (i, line) in enumerate(lines):
                if i == 0: continue
                label = line[-1]
                if label != target_label: continue
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[7]
                text_b = line[8]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                count += 1
                if count == k: break
        return examples


def input_example_to_tuple(example):
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]


def tokenize_multipart_input(
    input_text_list,
    max_length,
    label,
    tokenizer,
    template="*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
    first_sent_limit=None,
    other_sent_limit=None,
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token

    """
    Concatenate all sentences and prompts based on the provided template.
    Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
    *xx* represent variables:
        *cls*: cls_token
        *mask*: mask_token
        *sep*: sep_token
        *sep+*: sep_token, also means +1 for segment id
        *sent_i*: sentence i (input_text_list[i])
        *sent-_i*: same as above, but delete the last token
        *sentl_i*: same as above, but use lower case for the first word
        *sentl-_i*: same as above, but use lower case for the first word and delete the last token
        *+sent_i*: same as above, but add a space before the sentence
        *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
        *label_i*: label_word_list[i]
        *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

    Use "_" to replace space.
    PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
    """

    special_token_mapping = {
        'bos': tokenizer.bos_token_id, 'cls': tokenizer.cls_token_id, 'eos': tokenizer.eos_token_id, 
        'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id,
    }
    template_list = template.split('*') # Get variable list in the template
    segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

    for part_id, part in enumerate(template_list):
        new_tokens = []
        segment_plus_1_flag = False
        if part in special_token_mapping:
            new_tokens.append(special_token_mapping[part])
            if part == 'sep+':
                segment_plus_1_flag = True
        elif part[:6] == 'sent-_':
            # Delete the last token
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id][:-1])
        elif part[:6] == 'sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentl_':
            # Lower case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(' ' + text)
        else:
            # Just natural language prompt
            part = part.replace('_', ' ')
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer.convert_tokens_to_ids(part))
            else:
                new_tokens += enc(part)

        if part[:4] == 'sent' or part[1:5] == 'sent':
            # If this part is the sentence, limit the sentence length
            sent_id = int(part.split('_')[1])
            if sent_id == 0:
                new_tokens = new_tokens[:first_sent_limit]
            else:
                if other_sent_limit is not None:
                    new_tokens = new_tokens[:other_sent_limit]

        input_ids += new_tokens
        attention_mask += [1] * len(new_tokens)
        token_type_ids += [segment_id] * len(new_tokens)

        if segment_plus_1_flag:
            segment_id += 1

    # Padding
    ### Code below was originally commented out because they use dynamic padding rather than static padding to max_length
    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        # Default is to truncate the tail
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if tokenizer.mask_token_id is not None:
        # Make sure that the masked position is inside the max_length
        assert tokenizer.mask_token_id in input_ids, \
            "Mask token not found for input: {} {}".format(input_text_list, input_ids)
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        assert mask_pos[0] < max_length
    else:
        # autoregressive model
        mask_pos = [len(input_ids) - 1]

    result = {'input_ids': input_ids, 'attention_mask': attention_mask, 'mask_pos': mask_pos,
              'label': label}

    return result


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
mnli_processor = MnliProcessor()
snli_processor = SnliProcessor()

label_list = ["contradiction", "entailment", "neutral"]
num_labels = len(label_list)
label_to_word = {'contradiction': 'No', 'entailment': 'Yes', 'neutral': 'Maybe'}
label_map = {label: i for i, label in enumerate(label_list)}
num_sample = 1
template = "*cls**sent-_0*?*mask*,*+sentl_1**sep+*"

# get label_word_list
for key in label_to_word:
    # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
    if label_to_word[key][0] not in ['<', '[', '.', ',']:
        # Make sure space+word is in the vocabulary
        assert len(tokenizer.tokenize(' ' + label_to_word[key])) == 1
        label_to_word[key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + label_to_word[key])[0])
    else:
        label_to_word[key] = tokenizer.convert_tokens_to_ids(label_to_word[key])
    print("Label {} to word {} ({})".format(key, tokenizer.convert_ids_to_tokens(label_to_word[key]), label_to_word[key]))

label_word_list = [label_to_word[label] for label in label_list]
print("Label word list:", label_word_list)

train_examples = mnli_processor.get_train_examples(f"{os.path.dirname(__file__)}/original/MNLI", PRIVATE_SAMPLE_PER_CLASS)
test_examples = mnli_processor.get_dev_examples(f"{os.path.dirname(__file__)}/original/MNLI", 1000)
public_examples = snli_processor.get_train_examples(f"{os.path.dirname(__file__)}/original/SNLI", PUBLIC_SAMPLE_PER_CLASS)

print(f"Number of train examples: {len(train_examples)}")
print(f"Number of test examples: {len(test_examples)}")
print(f"Number of public examples: {len(public_examples)}")

features = {set_name: {'input_ids': [], 'attention_mask': [], 'mask_pos': [], 'label': []}
            for set_name in ["train", "test", "public"]}

for mode in ["train", "public", "test"]:
    for i, example in enumerate(eval(f"{mode}_examples")):
        tokenized_output = tokenize_multipart_input(
            input_text_list=input_example_to_tuple(example),
            max_length=MAX_LEN,
            label=label_map[example.label],
            tokenizer=tokenizer,
            template=template,
            first_sent_limit=first_sent_limit,
            other_sent_limit=other_sent_limit,
        )
        for attribute in ['input_ids', 'attention_mask', 'mask_pos', 'label']:
            features[mode][attribute].append(tokenized_output[attribute])
        
        # validate the first 3 examples for each mode
        if i < 3:
            text = tokenizer.decode(tokenized_output['input_ids'])
            print('mode', mode, 'text', tokenizer.decode(tokenized_output['input_ids']), 
                  'label', tokenized_output['label'])
    
    # Save features to disk
    cache_path = f"{os.path.dirname(__file__)}/mnli_snli_512/{mode}.pt"
    torch.save(features[mode], cache_path)


# validate the saved data
train_data = torch.load(f"{os.path.dirname(__file__)}/mnli_snli_512/train.pt")
test_data = torch.load(f"{os.path.dirname(__file__)}/mnli_snli_512/test.pt")
public_data = torch.load(f"{os.path.dirname(__file__)}/mnli_snli_512/public.pt")

for mode in ['train', 'test', 'public']:
    data = eval(f"{mode}_data")
    count = torch.tensor(data['label']).bincount(minlength=3)
    print(f"Number of {mode} examples: {len(data['input_ids'])}, "
          f"seq len: {len(data['input_ids'][0])}, "
          f"Class distribution: {count.tolist()}")


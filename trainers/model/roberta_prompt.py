"""
Custom models for few-shot learning specific operations.
Code is adapted from MEZO: https://github.com/princeton-nlp/MeZO
"""

import torch
import torch.nn.functional as F

from .modeling_roberta import RobertaPreTrainedModel, RobertaForMaskedLM


def model_for_prompting_forward(
    model,
    input_ids=None,
    attention_mask=None,
    mask_pos=None,
    labels=None,
):
    
    mask_pos = mask_pos.squeeze()

    # Encode everything
    outputs = model.roberta(input_ids, attention_mask=attention_mask)
    # Get <mask> token representation
    sequence_output = outputs[0]
    sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

    # Reshape mask_pos to [batch_size, 1, 1] so it can broadcast
    # mask_pos_exp = mask_pos.view(-1, 1, 1).expand(-1, 1, sequence_output.size(-1))

    # Use torch.gather to select the [MASK] embedding for each example
    # sequence_mask_output = torch.gather(sequence_output, dim=1, index=mask_pos_exp).squeeze(1)

    # Logits over vocabulary tokens
    prediction_mask_scores = model.lm_head(sequence_mask_output)

    # Return logits for each label
    logits = []
    for _id in model.label_word_list:
        logits.append(prediction_mask_scores[:, _id].unsqueeze(-1))
    logits = torch.cat(logits, -1)

    # loss =  nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

    if hasattr(model, "lr_weight"):
        # Linear head
        logits = torch.matmul(F.softmax(logits, -1), model.lr_weight) 
    if hasattr(model, "lr_bias"):
        logits += model.lr_bias.unsqueeze(0)

    output = logits
    return output
    # return ((loss,) + output) if loss is not None else output



class RobertaModelForPromptFinetuning(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.roberta = model.roberta
        self.lm_head = model.lm_head
        del model
        self.label_word_list = torch.tensor([440, 3216, 5359]).long().to(self.device)

    def forward(self, input_ids=None, attention_mask=None, mask_pos=None, labels=None):
        return model_for_prompting_forward(self, input_ids, attention_mask, mask_pos, labels)

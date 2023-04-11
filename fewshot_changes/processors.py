# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implements processors to convert examples to input and outputs, this can be
with integrarting patterns/verbalizers for PET or without."""
import abc 
import string 
from collections import OrderedDict

from .utils import Text, get_verbalization_ids, remove_final_punctuation, lowercase 


class AbstractProcessor(abc.ABC):
    def __init__(self, tokenizer, with_pattern, pattern_id=None, mask_position=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.mask_token_id
        self.mask_token = tokenizer.mask_token
        self.with_pattern = with_pattern 
        self.pattern_id = pattern_id
        self.tokenized_verbalizers = None  
        self.mask_position = mask_position

    def get_sentence_parts(self, example, mask_length):
        pass 

    def get_prompt_parts(self, example, mask_length):
         pass 

    def get_verbalizers(self):
        pass 

    def get_target(self, example):
        return example["fairness","non-moral","purity","degradation","loyalty","care","cheating","betrayal", "subversion", "authority", "harm"]

    def get_tokenized_verbalizers(self, example=None):
       """If verbalizers are fixed per examples, this returns back a computed tokenized 
       verbalizers, but if this is example dependent, it computes the tokenized verbalizers
       per example. In this function, as a default, we compute the static one."""
       if self.tokenized_verbalizers is not None:
            return self.tokenized_verbalizers

       verbalizers = self.get_verbalizers()
       assert len(verbalizers) != 0, "for using static tokenized verbalizers computation, the length"
       "of verbalizers cannot be empty."
       self.tokenized_verbalizers=[[get_verbalization_ids(word=verbalizer, tokenizer=self.tokenizer)] for verbalizer in verbalizers]
       return self.tokenized_verbalizers

    def get_extra_fields(self, example=None):
       # If there is a need to keep extra information, here we keep a dictionary
       # from keys to their values.
       return {} 

    def get_classification_parts(self, example):
          pass 
   
    def get_parts_with_setting_masks(self, part_0, part_1, masks):
        "Only used in case of two sentences: 0`: [p,h,m],[]  `1`: [p,m,h],[]  `2`: [p],[m,h] , `3`: [p],[h,m]"
        if self.mask_position == '0':
            return part_0+part_1+masks, []
        elif self.mask_position == '1':
            return part_0+masks+part_1, []
        elif self.mask_position == '2':
            return part_0, masks+part_1
        elif self.mask_position == '3':
            return part_0, part_1+masks 

class MFTC(AbstractProcessor):
    name = "mftc"

    def get_classification_parts(self, example):
        return example["text"], None

    def get_sentence_parts(self, example, mask_length):
        if not self.with_pattern:
            return [Text(text=example["text"], shortenable=True)]+mask_length*[Text(text=self.mask_token)], []
        return self.get_prompt_parts(example, mask_length)

    def get_prompt_parts(self, example, mask_length):
        source = Text(text=example["text"], shortenable=True)
        masks = mask_length*[Text(text=self.mask_token)]
        return [source, *masks], [example["fairness"], example["non-moral"], example["purity"], example["degradation"], example["loyalty"], example["care"], example["cheating"], example["betrayal"], example["subversion"], example["authority"], example["harm"]]

    def get_verbalizers(self):
        return ["0", "1"]
    



PROCESSOR_MAPPING = OrderedDict(
    [
        ('mftc', MFTC),
        # ('cr', CR),
        # ('subj', Subj),
        # ('trec', Trec),
        # ('SST-2', SST2),
        # ('sst-5', SST5),
        # #superglue datasets 
        # ('boolq', BoolQ),
        # ('rte', RTE),
        # ('cb', CB),
        # ('wic', WiC),
        # #glue datasets 
        # ('qnli', QNLI),
        # ('qqp', QQP),
        # ('mrpc', MRPC)
    ]
)

class AutoProcessor:
    @classmethod
    def get(self, task, tokenizer, with_pattern, pattern_id, mask_position):
        if task in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[task](
                tokenizer=tokenizer,
                with_pattern=with_pattern,
                pattern_id=pattern_id,
                mask_position=mask_position)
        raise ValueError(
            "Unrecognized task {} for AutoProcessor: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in PROCESSOR_MAPPING.keys())
            )
        )


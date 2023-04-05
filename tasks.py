# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import OrderedDict
from os.path import join 
from datasets import load_dataset, concatenate_datasets
import functools
import numpy as np 
import sys 
import torch 
from collections import Counter

from fewshot.metrics import metrics


class AbstractTask(abc.ABC):
    task = NotImplemented
    num_labels = NotImplemented

    def __init__(self, data_seed, num_samples, cache_dir, data_dir=None):
        self.data_seed = data_seed
        self.num_samples = num_samples 
        self.data_dir = data_dir 
        self.cache_dir = cache_dir

    def load_datasets(self):
        pass 
    
    def post_processing(self, datasets):
        return datasets 

    def sample_datasets(self, datasets):
        shuffled_train = datasets["train"].shuffle(seed=self.data_seed)
        
        if self.task in ["MFTC"]:
            datasets["test"] = datasets["validation"]

        if self.task in ["MFTC"]:
            # First filter, then shuffle, otherwise this results in a bug.
            # Samples `num_samples` elements from train as training and development sets.
            sampled_train = []
            sampled_dev = []
            for label in range(self.num_labels):
                data = shuffled_train.filter(lambda example: int(example['label']) == label)
                print(label, np.unique(data["label"]))
                num_samples = min(len(data)//2, self.num_samples)
                print(num_samples)
                sampled_train.append(data.select([i for i in range(num_samples)]))
                sampled_dev.append(data.select([i for i in range(num_samples, num_samples*2)]))

            # Joins the sampled data per label.
            datasets["train"] = concatenate_datasets(sampled_train)
            datasets["validation"] = concatenate_datasets(sampled_dev)
        return datasets

    def get_datasets(self):
        datasets = self.load_datasets()
        if self.num_samples is not None:
            datasets = self.sample_datasets(datasets)
            datasets = self.post_processing(datasets)
            label_distribution_train = Counter(datasets["train"]["label"])
            label_distribution_dev = Counter(datasets["validation"]["label"])
        return datasets 


# class MR(AbstractTask):
#     task = "mr"
#     num_labels = 2 
#     metric = [metrics.accuracy]

    # def load_datasets(self):
    #     dataset_args = {}
    #     print("task ", self.task)
    #     data_dir = join(self.data_dir, self.task) 
    #     data_files = {
    #         "train": join(data_dir, "train.json"),
    #         "test": join(data_dir, "test.json")
    #         }        
    #     return load_dataset("json", data_files=data_files, cache_dir=self.cache_dir, **dataset_args)


# class CR(MR):
#     task = "cr"
#     num_labels = 2 
#     metric = [metrics.accuracy]
   
# class Subj(MR):
#     task = "subj"
#     num_labels = 2 
#     metric = [metrics.accuracy]
   
class MFTC(AbstractTask):
    task = "mftc"
    num_labels = 9
    labels_list = ['0', '1','2', '3','4', '5','6', '7','8','9'] #["fairness","non-moral","purity","degradation","loyalty","care","cheating","betrayal","subversion","authority","harm"]
    metric = [metrics.accuracy]

    def load_datasets(self):
        dataset_args = {}
        print("task ", self.task)
        data_dir = join(self.data_dir, self.task) 
        data_files = {
            "train": join(data_dir, "train.json"),
            "test": join(data_dir, "test.json")
            }        
        return load_dataset("json", data_files=data_files, cache_dir=self.cache_dir, **dataset_args)


TASK_MAPPING = OrderedDict(
    [
        ('mftc', MFTC),
        # superglue datasets.
        ('mftc', MFTC),
        # glue datasets
        ('mftc', MFTC),
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, data_seed, num_samples, cache_dir, data_dir=None):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](
                data_seed=data_seed, 
                num_samples=num_samples, 
                cache_dir=cache_dir, 
                data_dir=data_dir)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )

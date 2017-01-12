# -*- coding: utf-8 -*-
# @Author: feidong1991
# @Date:   2016-10-17 11:15:40
# @Last Modified by:   feidong1991
# @Last Modified time: 2016-10-22 18:28:22
# Modified from max's code

import json
import os

# from utils import utils


class Alphabet:
    def __init__(self, name, keep_growing=True):
        self.__name = name

        self.instance2index = {}
        self.instances = []

        self.keep_growing = keep_growing

        # index 0 is occupied by default
        self.default_index = 0
        self.next_index = 1

        # self.logger = utils.get_logger("Alphabet")

    def add(self, instance):
        # add instance
        if instance not in self.instances:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        # get index for specific instance
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index

            else:
                return self.default_index

    def get_instance(self, index):
        # get instance for specific index
        if index == 0:
            # Index 0 is occupied by default
            return None
        try:
            return self.instances[index-1]
        except IndexError:
            self.logger.warn('unknown instance, return the first instance')
            return self.instances[0]

    def size(self):
        return len(self.instances) + 1

    def iteritems(self):
        return self.instance2index.iteritems()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is only allowed between [1: size of Alphbat]")
        return zip(range(start, len(self.instances)+1), self.instances[start-1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {'instance2index': self.instance2index, 'instances': self.instances}

    def from_json(self, data):
        self.instance2index = data['instance2index']
        self.instances = data['instances']

    def save(self, output_dir, name=None):

        saving_name = name if name else self.__name
        try:
            json.dump(self.get_content(), open(os.path.join(output_dir, saving_name + ".json"), 'w'))
        except Exception as e:
            self.logger.warn("Alphabet is not saved: " % repr(e))

    def load(self, input_dir, name=None):
        """
        Load json data
        """
        loading_name = name if name else self.__name
        self.from_json(json.load(open(os.path.join(input_dir, loading_name + ".json"))))
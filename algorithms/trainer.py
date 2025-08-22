'''
Abstract class for Trainer.

2025 Matt Taylor
'''

from abc import ABC, abstractmethod
import os

class Trainer(ABC):
    '''
    Abstract base class for any trainer.
    '''
    def __init__(self, cur_path=None, logger=None):
        self.cur_path = cur_path
        if cur_path is None:
            self.cur_path = os.path.dirname(os.path.realpath(__file__))
        self.logger = logger

    @abstractmethod
    def add_args(parser):
        '''
        Method for parsing arguments for the training class
        Args: 
            - parser: argparse object
        '''
        pass

    def get_init_kwargs(args):
        return {}

    @abstractmethod
    def get_training_kwargs(args):
        pass

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def load():
        pass

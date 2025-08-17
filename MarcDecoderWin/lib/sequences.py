from typing import Tuple, Dict
from math import ceil

import numpy as np
from tensorflow.keras.utils import Sequence as BaseSequence



class Sequence(BaseSequence):
    def __init__(
        self,
        rec_input: np.ndarray,
        eval_input: np.ndarray,
        log_errors: np.ndarray,
        batch_size: int,
    ) -> None:
        # print('seq shape', rec_input.shape, eval_input.shape, log_errors.shape)
        self.rec_input = rec_input
        self.eval_input = eval_input
        self.log_errors = log_errors

        self.seq_size = len(rec_input)
        self._num_batches = ceil(self.seq_size / batch_size)

        self._batch_size = batch_size

    def __len__(self) -> int:
        """
        __len__ Returns the number of batches per epoch

        Returns
        -------
        int
            Number of batches per epoch
        """
        return self._num_batches

    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        __getitem__ Returns a single batch of the dataset

        Returns
        -------
        tuple[dict[str, NDArray], NDArray]
            A tuple of a dictionary with the input data, consisting of the defects and
            final defects, together with the output label.
        """
        start_shot = index * self._batch_size
        if index == (self._num_batches - 1):
            end_shot = self.seq_size
        else:
            end_shot = start_shot + self._batch_size

        rec_input = self.rec_input[start_shot:end_shot]
        eval_input = self.eval_input[start_shot:end_shot]

        inputs = dict(rec_input=rec_input, eval_input=eval_input)

        log_errors = self.log_errors[start_shot:end_shot]

        return inputs, log_errors

    def values(self):
        inputs = dict(rec_input=self.rec_input, eval_input=self.eval_input)
        outputs = self.log_errors
        return inputs, outputs

import numpy as np

import logging
import torch
import copy
import os


class Checkpointer:

    def __init__(self, model, save_dir="", **checkpointables):
        '''
        Checkpointer saves configuration, evaluation results and model parameters
        to disk

        Args:
            model (nn.Module): Model
            save_dir (str): Checkpoint save directory
            checkpointables (dict): Evaluation metrics
        '''
        self.model = model
        self.checkpointables = copy.copy(checkpointables)
        self.logger = logging.getLogger(__name__)
        self.save_dir = save_dir

    def save(self, name: str, **kwargs: dict):
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, path: str):
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info(
                "No checkpoint found. Initializing model from scratch"
            )
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))

        checkpoint = self._load_file(path)
        self._load_model(checkpoint)
        for key, obj in self.checkpointables.items():
            if key in checkpoint:
                self.logger.info("Loading {} from {}".format(key, path))
                self.checkpointables[key] = checkpoint.pop(key)

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)

    def get_all_checkpoint_files(self):
        """
        Returns:
            list: All available checkpoint files (.pth files) in target
                directory.
        """
        all_model_checkpoints = [
            os.path.join(self.save_dir, file)
            for file in os.listdir(self.save_dir)
            if os.path.isfile(os.path.join(self.save_dir, file))
            and file.endswith(".pth")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool = True):
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
        return self.load(path)

    def tag_last_checkpoint(self, last_filename_basename: str):
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename_basename)

    def _load_file(self, f: str):
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.
        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        """
        Load weights from a checkpoint.
        Args:
            checkpoint (Any): checkpoint contains the weights.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # work around https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    self.logger.warning(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(
                            k, shape_checkpoint, shape_model
                        )
                    )
                    checkpoint_state_dict.pop(k)

        incompatible = self.model.load_state_dict(
            checkpoint_state_dict, strict=False
        )
        if incompatible.missing_keys:
            self.logger.info(
                "Missing keys: {}".format(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            self.logger.info(
                "Unexpected keys: {}".format(incompatible.unexpected_keys)
            )

    def _convert_ndarray_to_tensor(self, state_dict: dict):
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(
                v, torch.Tensor
            ):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(
                        k, type(v)
                    )
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)
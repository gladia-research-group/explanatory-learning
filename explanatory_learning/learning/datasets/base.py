import json
import random
from typing import Tuple

from explanatory_learning.data.encoders import *
from torch.utils.data import Dataset


class LabelingDataset(Dataset):
    """A sub-dataset of a given parent dataset, associated to a specific rule.

    This dataset view takes the inputs of a parent dataset associated only to a given
    rule.
    """

    def __init__(self, rule_id: int, parent_dataset: "ZendoDataset"):
        super().__init__()
        self.rule_id = rule_id
        self.parent_dataset = parent_dataset
        self.offset = parent_dataset.rule_id_to_idx(rule_id)

    @property
    def structures(self):
        return self.parent_dataset.structures

    @property
    def rules(self):
        return self.parent_dataset.rules

    def __len__(self):
        return self.parent_dataset.num_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)  # IMPORTANT
        return self.parent_dataset[self.offset + idx]


class ZendoDataset(Dataset):
    """Abstract Zendo dataset.

    This dataset loads information from a JSON data file, and digest it so it is
    easy to use. Override the methods *__getitem__* and *collate_fn* in its subclasses
    to implement specific behaviour.


    Each entry of this dataset is associated to a structure, a rule, the
    structure label w.r.t. the rule, and a pre-computed table. The association is many
    to one for the structures, meaning that multiple structures are associated to a
    single rule and table.

    :param json_file: name of JSON file containing the dataset data
    :param structure_length: length of the structures in input (default 6)
    :param num_samples: number of structures (out of all possible structures) sampled per rule
    :param rule_filter: boolean function that filters rules in the dataset
    """

    def __init__(self, json_file: str, num_samples: int = None, rule_filter=None):
        self.num_samples = num_samples

        if rule_filter is None:
            rule_filter = lambda x: True

        # load json
        with open(json_file, "r") as f:
            data = json.load(f)

        # structure dictionary
        self.structures = data["structures"]
        self._struct2id = {struct: i for i, struct in enumerate(self.structures)}
        structure_lengths = {len(struct) for struct in self.structures}
        assert len(structure_lengths) == 1
        self._structure_length = max(structure_lengths)

        self.structure_encoder = get_structure_encoder()
        encoded_structures = self.structure_encoder.transform(self.structures).t()
        self._encoded_structures = encoded_structures

        # loop variables
        self.rules = []
        self._rule2id = dict()
        self._tables = []
        self._table_labels = []
        self._labels = []
        self._struct_ids = []

        # _tmp_range = list(range(len(self.structures)))
        _tmp_ones = torch.ones([len(self.structures)])

        for line_id, line in enumerate(data["dataset"]):

            rule = line["rule"]
            table = line["table"]
            table_labels = line["table_labels"]
            labels = line["labels"]

            if not rule_filter(rule.split()):
                continue

            rule_id = len(self.rules)

            self.rules.append(rule)
            self._rule2id[rule] = rule_id

            if self.num_samples is None:
                self.num_samples = len(labels)

            # get random sample
            # struct_ids_sample_tmp = sorted(random.sample(_tmp_range, k=self.num_samples))
            _samples = torch.multinomial(_tmp_ones, num_samples=self.num_samples).view(
                -1
            )
            struct_ids_sample_tmp = torch.sort(_samples, dim=-1, descending=False)[0]
            del _samples

            # create table ids
            table_ids = []
            for structure in table:
                struct_id = self.id_from_struct(structure)
                table_ids.append(struct_id)

            # create structures
            structure_ids = struct_ids_sample_tmp
            structure_labels = torch.tensor(labels, dtype=torch.bool)

            # convert to torch array
            table_ids = torch.tensor(table_ids, dtype=torch.long)
            table_labels = torch.tensor(table_labels, dtype=torch.long)

            # update variables
            self._tables.append(table_ids)
            self._table_labels.append(table_labels)
            self._struct_ids.append(structure_ids)
            self._labels.append(structure_labels)

            # free resources
            del table_labels, table_ids, struct_ids_sample_tmp

    def __len__(self) -> int:
        """Number of entries in the dataset.

        :return: number of entries in the dataset.
        """
        return len(self._labels) * self.num_samples

    # indexing functions ----------------------------------------------------------
    def idx_to_rule_id(self, idx: int) -> int:
        """Convert from dataset index to the associated rule-id.

        Rule-ids are a fast way to identify rules and are used internally to
        index rules.
        """
        rule_id = idx // self.num_samples
        return rule_id

    def rule_id_to_idx(self, rule_id: int) -> int:
        """Converts from a rule-id to the first associated dataset entry index.

        To a single rule are associated multiple structures (equal to the number
        specified in the constructor). These are contiguous in the index space of
        the dataset, and the first one is returned by this method.
        """
        return rule_id * self.num_samples

    def idx_to_struct_id(self, idx: int) -> int:
        """Converts from a dataset index to the associated structure-id.

        Structures-ids are a fast way to identify structures and are used internally to
        index quickly.
        """
        rule_id = self.idx_to_rule_id(idx)
        struct_idx = idx % self.num_samples

        struct_id = self._struct_ids[rule_id][struct_idx]
        return struct_id

    def idx_to_label(self, idx: int) -> int:
        """Converts from a dataset index to the associated Zendo tag."""
        rule_id = self.idx_to_rule_id(idx)
        struct_id = self.idx_to_struct_id(idx)
        label = self._labels[rule_id][struct_id]
        return label

    def rule_from_id(self, rule_id: int) -> str:
        """Converts a rule-id to the corresponding rule."""
        rule_id = self.rules[rule_id]
        return rule_id

    def struct_from_id(self, struct_id: int) -> str:
        "Converts a structure-id to the corresponding structure."
        rule_id = self.structures[struct_id]
        return rule_id

    def id_from_struct(self, struct: str) -> int:
        "Converts a structure to the corresponding structure-id."
        return self._struct2id[struct]

    def id_from_rule(self, rule: str) -> int:
        "Converts a rule to the corresponding rule-id."
        return self._rule2id[rule]

    def rule_labeling(self, rule_id: int) -> Dataset:
        "Returns a sub-dataset composed only by the entries associated to the input rule."
        return LabelingDataset(rule_id=rule_id, parent_dataset=self)

    # functions to implement --------------------------------------------------
    def collate_fn(self, input_list) -> Tuple:
        """Collate function specific for this dataset.

        This function should be passed as input to a dataloader in order to prepare
        the batch data.

        :param input_list: list of tuples containing the entries of the dataset associated
            to a batch.

        :return: a tuple of tensors representing the batch data.
        """
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Tuple:
        raise NotImplementedError()


# ----------------------------------------------------------------------------------------


class FalsifierDataset(ZendoDataset):
    """
    Dataset with rules and tagged structures.

    This dataset is used to train the Falsifier model; the entries in the dataset
    are tuples of type (rule, structure, tag).

    .. note::
        The custom collate function ``collate_fn`` should be passed to the dataloader in
        order to correctly format the data.

    :param json_file: name of JSON file containing the dataset data
    :param num_samples: number of structures (out of all possible structures) sampled per rule
    :param rule_filter: boolean function that filters rules in the dataset
    :param rule_encoder: object that tokenizes and index rules in the dataset
    """

    def __init__(
        self, json_file: str, num_samples: int = None, rule_filter=None, **kwargs
    ):

        super().__init__(
            json_file=json_file, num_samples=num_samples, rule_filter=rule_filter
        )

        for k in kwargs.keys():
            warn(f"ignored named parameter {k}!")

        self.rule_encoder = get_rule_encoder()
        self._encoded_rules = self.rule_encoder.transform(self.rules).t()

    def __len__(self) -> int:
        """Number of entries in the dataset.

        :return: number of entries in the dataset.
        """
        return len(self._labels) * self.num_samples

    def __getitem__(self, idx: int) -> Tuple:
        rule_id = self.idx_to_rule_id(idx)
        struct_id = self.idx_to_struct_id(idx)
        label = self.idx_to_label(idx)
        return rule_id, struct_id, label

    def collate_fn(self, input_list) -> Tuple:

        # unpack batch
        rule_ids = [ri for ri, si, l in input_list]
        struct_ids = [si for _, si, _ in input_list]
        labels = [label for _, _, label in input_list]

        rule_ids = torch.tensor(rule_ids, dtype=torch.long)
        rules = self._encoded_rules.index_select(dim=0, index=rule_ids).t_()

        labels = torch.tensor(labels, dtype=torch.long)

        struct_ids = torch.tensor(struct_ids, dtype=torch.long)

        # NOTE: important when batch < 32
        structures = self._encoded_structures.index_select(dim=0, index=struct_ids).t_()

        out = (rules, structures, labels)
        return out

    def prepare_data(self, rule: str, structures: List[str]) -> Tuple:
        raise NotImplementedError()


class EmpiricistDataset(ZendoDataset):
    """
    Dataset with tables and tagged structures.

    This dataset is used to work on the Empiricist model; the entries in the dataset
    are tuples of type (table, table_tags, structure, tag).

    .. note::
        The custom collate function ``collate_fn`` should be passed to the dataloader in
        order to correctly format the data.

    :param json_file: name of JSON file containing the dataset data
    :param num_samples: number of structures (out of all possible structures) sampled per rule
    :param rule_filter: boolean function that filters rules in the dataset
    """

    def __init__(self, json_file: str, num_samples: int = None, rule_filter=None):
        super().__init__(
            json_file=json_file, num_samples=num_samples, rule_filter=rule_filter
        )

    def __getitem__(self, idx: int) -> Tuple:
        rule_id = self.idx_to_rule_id(idx)
        struct_id = self.idx_to_struct_id(idx)
        label = self.idx_to_label(idx)

        table = self._tables[rule_id]
        table_labels = self._table_labels[rule_id]

        return table, table_labels, struct_id, label

    def collate_fn(self, input_list) -> Tuple:
        # get sizes
        batch_size = len(input_list)

        # NOTE: assuming same size across all tables in batch!!!
        table_size = len(input_list[0][0])
        structure_length = self._structure_length

        # pre-allocate tensors
        tables = torch.zeros(
            size=[table_size, batch_size, structure_length], dtype=torch.long
        )
        table_labels = torch.zeros(size=[table_size, batch_size], dtype=torch.long)
        structures = torch.zeros(size=[structure_length, batch_size], dtype=torch.long)
        labels = torch.zeros(size=[batch_size], dtype=torch.long)

        for batch_id, (t, tlabels, struct_id, label) in enumerate(input_list):
            tables[:, batch_id, :] = self._encoded_structures[t]
            table_labels[:, batch_id] = tlabels
            labels[batch_id] = label
            structures[:, batch_id] = self._encoded_structures[struct_id]

        out = (tables, table_labels, structures, labels)
        return out


class RandomizedEmpiricistDataset(EmpiricistDataset):
    """
    Dataset with tagged structures and randomized tables.

    This dataset is used to train the Empiricist model; the entries in the dataset
    are tuples of type (table, table_tags, structure, tag).
    Differently from its superclass, the table is chosen at random. This can be helpful in
    order to avoid table memorization while training.

    .. note::
        The custom collate function ``collate_fn`` should be passed to the dataloader in
        order to correctly format the data.

    :param json_file: name of JSON file containing the dataset data
    :param num_samples: number of structures (out of all possible structures) sampled per rule
    :param rule_filter: boolean function that filters rules in the dataset
    :param table_size: size of the random table
    """

    def __init__(
        self,
        json_file: str,
        num_samples: int = None,
        rule_filter=None,
        table_size: int = 32,
    ):
        super().__init__(
            json_file=json_file, num_samples=num_samples, rule_filter=rule_filter
        )

        self.table_size = table_size
        self._random_indices = torch.randint(
            low=0, high=self.num_samples, size=[len(self.structures) * 30]
        )

    def __getitem__(self, idx: int) -> Tuple:
        rule_id = self.idx_to_rule_id(idx)
        rule_labels_start = self.rule_id_to_idx(rule_id)

        N = self._random_indices.size(0)
        table_start = random.randint(0, N - self.table_size)
        table_end = table_start + self.table_size

        I = self._random_indices[table_start:table_end]
        table = self._struct_ids[rule_id][I]

        struct_id = self.idx_to_struct_id(idx)
        struct_label = self.idx_to_label(idx)
        table_labels = self._labels[rule_id][table]

        return table, table_labels, struct_id, struct_label


# -------------------------------------------------------------------------------


class AwareEmpiricistDataset(ZendoDataset):
    """
    Dataset with rules, tables, and tagged structures.

    This dataset is used to work on the Aware Empiricist model; the entries in the dataset
    are tuples of type (table, table_tags, structure, tag, rule).

    .. note::
        The custom collate function ``collate_fn`` should be passed to the dataloader in
        order to correctly format the data.

    :param json_file: name of JSON file containing the dataset data
    :param structure_length: length of the structures in input (default 6)
    :param num_samples: number of structures (out of all possible structures) sampled per rule
    :param rule_filter: boolean function that filters rules in the dataset
    :param rule_encoder: object that tokenizes and index rules in the dataset.
    """

    def __init__(
        self,
        json_file: str,
        num_samples: int = None,
        rule_filter=None,
        rule_encoder: RuleEncoder = None,
    ):
        super().__init__(
            json_file=json_file, num_samples=num_samples, rule_filter=rule_filter
        )

        self.rule_encoder = get_rule_encoder()
        self._encoded_rules = self.rule_encoder.transform(self.rules).t()

    def __getitem__(self, idx: int) -> Tuple:
        rule_id = self.idx_to_rule_id(idx)
        struct_id = self.idx_to_struct_id(idx)
        label = self.idx_to_label(idx)

        table = self._tables[rule_id]
        table_labels = self._table_labels[rule_id]

        return table, table_labels, struct_id, label, rule_id

    def collate_fn(self, input_list) -> Tuple:
        # get sizes
        batch_size = len(input_list)

        # NOTE: assuming same size across all tables in batch!!!
        table_size = len(input_list[0][0])
        structure_length = self._structure_length

        rule_ids = torch.tensor([ri for _, _, _, _, ri in input_list], dtype=torch.long)
        labels = torch.tensor([l for _, _, _, l, _ in input_list], dtype=torch.long)
        struct_ids = torch.tensor(
            [si for _, _, si, _, _ in input_list], dtype=torch.long
        )

        rules = self._encoded_rules.index_select(dim=0, index=rule_ids).t_()
        structures = self._encoded_structures.index_select(dim=0, index=struct_ids).t_()

        # pre-allocate tensors
        tables = torch.zeros(
            size=[table_size, batch_size, structure_length], dtype=torch.long
        )
        table_labels = torch.zeros(size=[table_size, batch_size], dtype=torch.long)

        for batch_id, (t, tlabels, struct_id, label, _) in enumerate(input_list):
            tables[:, batch_id, :] = self._encoded_structures[t]
            table_labels[:, batch_id] = tlabels

        out = (tables, table_labels, structures, labels, rules)
        return out


class RandomizedAwareDataset(AwareEmpiricistDataset):
    """
    Dataset with rules, randomized tables, and tagged structures.

    This dataset is used to train the Aware Empiricist model; the entries in the dataset
    are tuples of type (table, table_tags, structure, tag, rule).
    Differently from its superclass, the table is chosen at random. This can be helpful in
    order to avoid table memorization while training.

    .. note::
        The custom collate function ``collate_fn`` should be passed to the dataloader in
        order to correctly format the data.

    :param json_file: name of JSON file containing the dataset data
    :param structure_length: length of the structures in input (default 6)
    :param num_samples: number of structures (out of all possible structures) sampled per rule
    :param rule_filter: boolean function that filters rules in the dataset
    :param table_size: size of the random table
    :param rule_encoder: object that tokenizes and index rules in the dataset.
    """

    def __init__(
        self,
        json_file: str,
        num_samples: int = None,
        table_size: int = 32,
        rule_filter=None,
        rule_encoder: RuleEncoder = None,
    ):
        super().__init__(
            json_file=json_file,
            num_samples=num_samples,
            rule_filter=rule_filter,
            rule_encoder=rule_encoder,
        )

        self.table_size = table_size
        self._random_indices = torch.randint(
            low=0, high=self.num_samples, size=[len(self.structures) * 30]
        )

    def __getitem__(self, idx: int) -> Tuple:
        rule_id = self.idx_to_rule_id(idx)

        N = self._random_indices.size(0)
        table_start = random.randint(0, N - self.table_size)
        table_end = table_start + self.table_size

        I = self._random_indices[table_start:table_end]
        table = self._struct_ids[rule_id][I]

        struct_id = self.idx_to_struct_id(idx)
        struct_label = self.idx_to_label(idx)
        table_labels = self._labels[rule_id][table]

        return table, table_labels, struct_id, struct_label, rule_id

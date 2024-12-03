import torch
import os
import pickle
import numpy as np
from captum.concept import Concept
from eegatscale.transforms import Standardize, StandardizeLabel

class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=Standardize(), return_label=False, verbose=True, remove_last_channel=False):
        self.directory = directory
        self.transform = transform
        self.label_transform = StandardizeLabel()
        self.return_label = return_label
        self.verbose = verbose
        self.remove_last_channel = remove_last_channel
        self.paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]


    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f:
            data = pickle.load(f)
        if self.remove_last_channel:
            # remove last channel
            data = data[:, :-1,:]
        data = data.squeeze()
        # make the data as torch 32 bit float
        data = torch.tensor(data, dtype=torch.float32)
        if self.transform:
            data = self.transform(data)
        if self.return_label:
            label = self.get_label(idx)
            return data, label
        return data
    
    def get_label(self, idx):
        description = self.paths[idx].split("_")[-1].split(".")[0]
        label = 0 if description == "T1" else 1
        return label


# # get x amount of subsets of given size y from the dataset
# def get_subsets(dataset, n_samples, n_subsets):
#     subsets = []
#     for i in range(n_subsets):
#         indices = torch.randperm(len(dataset))[:n_samples]
#         subset = torch.utils.data.Subset(dataset, indices)
#         subsets.append(subset)
#     return subsets

# do the same as above just with numpy where i can specify the seed
def get_subsets_numpy(dataset, n_samples:int, n_subsets:int):
    subsets = []
    if n_samples > len(dataset):
        print(f"number of samples ({n_samples}) cant be greater than the dataset size ({len(dataset)}). Setting n_samples to dataset size")
        n_samples = len(dataset)
    for i in range(n_subsets):
        rng = np.random.default_rng(i)
        indices = rng.choice(len(dataset), n_samples, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
        subsets.append(subset)
    return subsets



def get_concepts_numpy(dataset, concept_name, concept_sample_size:int, n_concepts:int, idx_start:int=0, batch_size:int=1):
    concepts = []
    if concept_sample_size > len(dataset):
        print(f"number of samples ({concept_sample_size}) cant be greater than the dataset size ({len(dataset)}). Setting concept_sample_size to dataset size")
        concept_sample_size = len(dataset)
    for i in range(n_concepts):
        rng = np.random.default_rng(i)
        indices = rng.choice(len(dataset), concept_sample_size, replace=False)
        # indices = torch.randperm(len(dataset))[:concept_sample_size]
        concept_data = torch.utils.data.Subset(dataset, indices)
        concept_dataloader = torch.utils.data.DataLoader(concept_data, batch_size=batch_size)
        concept = Concept(id=idx_start+i, name=f"{concept_name}_{i:03d}", data_iter=concept_dataloader)
        concepts.append(concept)
    return concepts


"""
one experiment is essentially an experimental set

so for example

{left_fist_concept, random_eeg} is an experiment, and we will create it using the class
{right_fist_concept, random_eeg} is an experiment

"""


class Experiment:
    def __init__(self, model, inputs_name, concept_dir_path, concept_dir_names,  n_runs:int = 100, n_concept_samples:int = 75, verbose:bool = False, internal_concept_idx:int = 0, batch_size:int = 1):
        """
        
        Args:
            model: the model that we are using
            concept_dir_path: the path to the directory containing the concept datasets
            concept_dir_names: the names of the concept dirs that are located in the concept_dir_path
            n_runs: the number of runs we are doing (we are sampling n_concept_samples from each concept dataset n_runs times)
            n_concept_samples: the number of samples we are taking from each concept dataset


        """
        self.model = model
        self.concept_names = concept_dir_names
        self.concept_dir_path = concept_dir_path
        self.n_runs = n_runs
        self.n_concept_samples = n_concept_samples
        self.batch_size = batch_size
        self.inputs_name = inputs_name
        self.concepts = {}
        self.datasets: dict[str, PickleDataset] = {}
        self.verbose = verbose
        self.internal_concept_idx = internal_concept_idx

        self.load_datasets()
        self.load_concepts()
        self.inputs = next(iter(self.get_input_dataloader()))

    def load_datasets(self):
        for concept_name in self.concept_names:
            concept_path = os.path.join(self.concept_dir_path, concept_name)
            concept_dataset = PickleDataset(concept_path)
            self.datasets[concept_name] = concept_dataset

            if self.verbose:
                print(f"Loaded dataset {concept_name} with {len(concept_dataset)} samples")
        
        self.input_dataset = PickleDataset(os.path.join(self.concept_dir_path, self.inputs_name))
        if self.verbose:
            print(f"Loaded input dataset {self.inputs_name} with {len(self.input_dataset)} samples")

    def load_concepts(self):
        for concept_name in self.concept_names:
            # if we are doing 100 runs we need a 100 subsets of the dataset, randomly sampled
            concepts = get_concepts_numpy(self.datasets[concept_name], concept_name=concept_name, concept_sample_size=self.n_concept_samples, n_concepts=self.n_runs, idx_start=self.internal_concept_idx, batch_size=self.batch_size)
            self.concepts[concept_name] = concepts
            self.internal_concept_idx += len(concepts)


    def get_input_dataloader(self):
        # input_dataloader = torch.utils.data.DataLoader(self.input_dataset, batch_size=self.n_concept_samples*2)
        input_dataloader = torch.utils.data.DataLoader(self.input_dataset, batch_size=self.n_concept_samples*2)
        return input_dataloader
    

    @property
    def experimental_sets(self):
        experimental_sets = []
        for i in range(self.n_runs):
            experimental_set = [self.concepts[concept_name][i] for concept_name in self.concept_names]
            experimental_sets.append(experimental_set)
        return experimental_sets


import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, inputs, targets, input_encoder, target_encoder, block_size, input_pad_idx, target_pad_idx):
        self.inputs = inputs
        self.targets = targets
        self.input_encoder = input_encoder
        self.target_encoder = target_encoder
        self.block_size = block_size
        self.input_pad_idx = input_pad_idx
        self.target_pad_idx = target_pad_idx
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        # Tokenize the input and target sequences
        input_tokens = self.input_encoder(self.inputs[index])
        target_tokens = self.target_encoder(self.targets[index])
        
        # Pad or truncate the input and target sequences to a fixed length
        input_tokens = input_tokens[:self.block_size] + [self.input_pad_idx] * (self.block_size - len(input_tokens))
        target_tokens = target_tokens[:self.block_size] + [self.target_pad_idx] * (self.block_size - len(target_tokens))
        
        # Create attention masks
        input_mask = [1 if token != self.input_pad_idx else 0 for token in input_tokens]
        target_mask = [1 if token != self.target_pad_idx else 0 for token in target_tokens]
        
        # Convert the input and target sequences to PyTorch tensors
        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        target_tensor = torch.tensor(target_tokens, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        target_mask = torch.tensor(target_mask, dtype=torch.long)
        
        return (input_tensor, input_mask, target_tensor, target_mask)
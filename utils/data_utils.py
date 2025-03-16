import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import random
from PIL import Image

class UCF101PatchedDataset(Dataset):
    """Dataset for patched UCF101 images with improved preprocessing"""
    
    def __init__(self, 
                 data_path,
                 batch_idx,
                 patch_size=80,
                 num_patches_h=3,
                 num_patches_w=4,
                 transform=None,
                 augment=False):
        
        self.data_path = data_path
        self.transform = transform
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.augment = augment
        
        # Load the specific batch file
        batch_file = f"ucf101_subset_batch_{batch_idx}.npy"
        full_path = os.path.join(data_path, batch_file)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Batch file {full_path} not found")
        
        # Load the data - shape should be (n_videos, n_frames, height, width, channels)
        self.data = np.load(full_path)
        print(f"Loaded batch {batch_idx} with shape: {self.data.shape}")
        
        # Calculate total number of frames across all videos
        self.n_videos = self.data.shape[0]
        self.n_frames = self.data.shape[1]
        self.total_frames = self.n_videos * self.n_frames
        
        # For UCF101 images are 240x320 in shape
        # Calculate actual patch sizes
        self.actual_h_patch_size = self.data.shape[2] // self.num_patches_h
        self.actual_w_patch_size = self.data.shape[3] // self.num_patches_w
        
        print(f"Using patch sizes of {self.actual_h_patch_size}x{self.actual_w_patch_size}")
        
        # Set up augmentation transforms if enabled
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            ])
        
    def __len__(self):
        return self.total_frames * self.num_patches_h * self.num_patches_w
    
    def __getitem__(self, idx):
        # Map the patch index to video, frame, and patch position
        patches_per_frame = self.num_patches_h * self.num_patches_w
        total_patches_per_video = self.n_frames * patches_per_frame
        
        video_idx = idx // total_patches_per_video
        frame_idx = (idx % total_patches_per_video) // patches_per_frame
        patch_idx = idx % patches_per_frame
        
        patch_h = patch_idx // self.num_patches_w
        patch_w = patch_idx % self.num_patches_w
        
        # Ensure we stay within bounds
        video_idx = min(video_idx, self.n_videos - 1)
        frame_idx = min(frame_idx, self.n_frames - 1)
        
        # Get the full frame
        frame = self.data[video_idx, frame_idx]
        
        # Extract the patch
        h_start = patch_h * self.actual_h_patch_size
        w_start = patch_w * self.actual_w_patch_size
        h_end = min(h_start + self.actual_h_patch_size, frame.shape[0])
        w_end = min(w_start + self.actual_w_patch_size, frame.shape[1])
        
        patch = frame[h_start:h_end, w_start:w_end]
        
        # Convert patch to float and normalize to [0, 1]
        patch = patch.astype(np.float32) / 255.0
        
        # Convert to tensor with shape [C, H, W]
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        
        # Apply data augmentation if enabled
        if self.augment:
            patch = self.aug_transform(patch)
        
        # Apply additional transformations if provided
        if self.transform:
            patch = self.transform(patch)
            
        # Create metadata for potential visualization
        metadata = {
            'video_idx': video_idx,
            'frame_idx': frame_idx,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
            
        return patch, metadata

class VideoSequenceDataset(Dataset):
    """Dataset that returns sequences of frames for temporal modeling"""
    
    def __init__(self, 
                 data_path,
                 batch_idx,
                 sequence_length=8,
                 patch_size=80,
                 num_patches_h=3,
                 num_patches_w=4,
                 transform=None,
                 augment=False):
        
        self.data_path = data_path
        self.transform = transform
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.augment = augment
        
        # Load the specific batch file
        batch_file = f"ucf101_subset_batch_{batch_idx}.npy"
        full_path = os.path.join(data_path, batch_file)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Batch file {full_path} not found")
        
        # Load the data - shape should be (n_videos, n_frames, height, width, channels)
        self.data = np.load(full_path)
        print(f"Loaded batch {batch_idx} with shape: {self.data.shape}")
        
        # Calculate total number of valid sequences across all videos
        self.n_videos = self.data.shape[0]
        self.n_frames = self.data.shape[1]
        
        # For UCF101 images are 240x320 in shape
        # Calculate actual patch sizes
        self.actual_h_patch_size = self.data.shape[2] // self.num_patches_h
        self.actual_w_patch_size = self.data.shape[3] // self.num_patches_w
        
        # Calculate total number of valid sequences (allowing overlap)
        self.valid_sequence_starts = []
        for video_idx in range(self.n_videos):
            for start_frame in range(self.n_frames - self.sequence_length + 1):
                for patch_h in range(self.num_patches_h):
                    for patch_w in range(self.num_patches_w):
                        self.valid_sequence_starts.append((video_idx, start_frame, patch_h, patch_w))
        
        print(f"Dataset contains {len(self.valid_sequence_starts)} valid sequences")
        
        # Set up augmentation transforms if enabled
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            ])
    
    def __len__(self):
        return len(self.valid_sequence_starts)
    
    def __getitem__(self, idx):
        # Get sequence info
        video_idx, start_frame, patch_h, patch_w = self.valid_sequence_starts[idx]
        
        # Extract sequence of patches
        sequence = []
        for frame_offset in range(self.sequence_length):
            frame_idx = start_frame + frame_offset
            
            # Get the full frame
            frame = self.data[video_idx, frame_idx]
            
            # Extract the patch
            h_start = patch_h * self.actual_h_patch_size
            w_start = patch_w * self.actual_w_patch_size
            h_end = min(h_start + self.actual_h_patch_size, frame.shape[0])
            w_end = min(w_start + self.actual_w_patch_size, frame.shape[1])
            
            patch = frame[h_start:h_end, w_start:w_end]
            
            # Convert patch to float and normalize to [0, 1]
            patch = patch.astype(np.float32) / 255.0
            
            # Convert to tensor with shape [C, H, W]
            patch = torch.from_numpy(patch).permute(2, 0, 1).float()
            
            # Apply transforms as needed
            if self.transform:
                patch = self.transform(patch)
                
            sequence.append(patch)
        
        # Stack sequence along batch dimension then combine
        sequence_tensor = torch.stack(sequence)  # [T, C, H, W]
        
        # Apply same augmentation to all frames in sequence if enabled
        if self.augment:
            # Create a consistent augmentation for the whole sequence
            if random.random() > 0.5:  # horizontal flip
                sequence_tensor = torch.flip(sequence_tensor, dims=[3])  # flip width dim
            
            # Apply color jitter consistently across the sequence
            if random.random() > 0.5:
                brightness = 1.0 + random.uniform(-0.05, 0.05)
                contrast = 1.0 + random.uniform(-0.05, 0.05)
                saturation = 1.0 + random.uniform(-0.05, 0.05)
                sequence_tensor = transforms.functional.adjust_brightness(sequence_tensor, brightness)
                sequence_tensor = transforms.functional.adjust_contrast(sequence_tensor, contrast)
                sequence_tensor = transforms.functional.adjust_saturation(sequence_tensor, saturation)
        
        # Create metadata for potential visualization
        metadata = {
            'video_idx': video_idx,
            'start_frame': start_frame,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
            
        return sequence_tensor, metadata

def get_data_loaders(data_path, batch_idx, batch_size=32, patch_size=80, 
                     num_patches_h=3, num_patches_w=4, num_workers=2, 
                     val_split=0.1, is_test_set=False, use_sequences=False,
                     sequence_length=8, augment=True):
    """Create data loaders for a specific batch file with improved options"""
    
    # Base transform - normalization will be handled in the model
    transform = None
    
    # Choose the appropriate dataset class
    if use_sequences:
        dataset_class = VideoSequenceDataset
        # Adjust dataset parameters for sequences
        dataset_params = {
            'data_path': data_path,
            'batch_idx': batch_idx,
            'sequence_length': sequence_length,
            'patch_size': patch_size,
            'num_patches_h': num_patches_h,
            'num_patches_w': num_patches_w,
            'transform': transform,
            'augment': augment and not is_test_set  # No augmentation for test set
        }
    else:
        dataset_class = UCF101PatchedDataset
        # Regular dataset parameters
        dataset_params = {
            'data_path': data_path,
            'batch_idx': batch_idx,
            'patch_size': patch_size,
            'num_patches_h': num_patches_h,
            'num_patches_w': num_patches_w,
            'transform': transform,
            'augment': augment and not is_test_set  # No augmentation for test set
        }
    
    # Create dataset
    dataset = dataset_class(**dataset_params)
    
    # For test set, don't split
    if is_test_set:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False if num_workers == 0 else True
        )
        return test_loader
    
    # Split into train/val
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders with memory-efficient settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False if num_workers == 0 else True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False if num_workers == 0 else True,
        drop_last=False
    )
    
    return train_loader, val_loader

def get_all_batch_files(data_path, exclude_test=None):
    """Get list of all batch files except the test set if specified"""
    batch_files = glob.glob(os.path.join(data_path, "ucf101_subset_batch_*.npy"))
    batch_indices = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in batch_files]
    
    # Exclude test set if specified
    if exclude_test is not None:
        batch_indices = [idx for idx in batch_indices if idx != exclude_test]
    
    return sorted(batch_indices)
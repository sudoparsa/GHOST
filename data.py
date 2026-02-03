import json
from torchvision.datasets.folder import default_loader
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image


class COCO(Dataset):
    def __init__(self, coco_dir, split='train', transform=None):
        self.image_dir = os.path.join(coco_dir, f"images/{split}2017/")
        with open(os.path.join(coco_dir, f"annotations/instances_{split}2017.json"), 'r') as file:
            coco = json.load(file)
        with open(os.path.join(coco_dir, f"annotations/captions_{split}2017.json"), 'r') as file:
            self.coco_captions = json.load(file)
        
        self.transform = transform
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}

        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']] = ann
        
        for img in coco['images']:
            self.im_dict[img['id']] = img
        
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat

        
    def __len__(self):
        return len(list(self.im_dict.keys()))
    
    def __getitem__(self, idx):
        img = self.im_dict[idx]
        path = os.path.join(self.image_dir, img['file_name'])
        image = default_loader(path)
        if self.transform is not None:
            image = image.resize(self.transform, Image.LANCZOS)

        return image, path
        
        
    def get_targets(self, idx):
        return [self.cat_dict[ann['category_id']]['name'] for ann in self.annIm_dict[idx]]
    
    def get_bounding_boxes(self, idx):
        return [(self.cat_dict[ann['category_id']]['name'], ann['bbox'])for ann in self.annIm_dict[idx]]
    
    def get_captions(self, idx):
        caps = []
        for ann in self.coco_captions['annotations']:
            if ann['image_id'] == idx:
                caps.append(ann['caption'])
        return caps
    
    def get_categories(self, supercategory):
        return [self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys() if self.cat_dict[cat_id]['supercategory']==supercategory]
    

    def get_all_supercategories(self):
        return {self.cat_dict[cat_id]['supercategory'] for cat_id in self.cat_dict.keys()}
    
    def get_spurious_supercategories(self):
        return ['kitchen', 'food', 'vehicle',
                'furniture', 'appliance', 'indoor',
                'outdoor', 'electronic', 'sports',
                'accessory', 'animal']
    
    def get_no_classes(self, supercategories):
        return len([self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys() if self.cat_dict[cat_id]['supercategory'] in supercategories])
    

    def get_imgIds(self):
        return list(self.im_dict.keys())
    
    def get_all_targets_names(self):
        return [self.cat_dict[cat_id]['name'] for cat_id in self.cat_dict.keys()]
    
    def get_imgIds_by_class(self, present_classes=[], absent_classes=[]):
        # Return images that has at least one of the present_classes, and none of the absent_classes
        ids = []
        for img_id in self.get_imgIds():
            targets = self.get_targets(img_id)
            flag = False
            for c in present_classes:
                if c in targets:
                    flag = True
                    break
            for c in absent_classes:
                if c in targets:
                    flag = False
                    break
            if flag:
                ids.append(img_id)
        return ids



class NegativeImageFolder(Dataset):
    """
    A dataset for negative samples, structured as root/object_name/image.png.
    It returns a PIL image and the corresponding object_name.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the object subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.loader = default_loader
        self.samples = []
        
        # Get all class names from the subdirectories and sort them
        self.object_names = ["traffic light", "carrot", "toilet","knife","bottle","vase","clock","bus","boat","suitcase"]

        for object_name in self.object_names:
            object_dir = self.root_dir.format(cls=object_name)
            for img_file in sorted(os.listdir(object_dir)):
                # Ensure we are only picking up image files
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    path = os.path.join(object_dir, img_file)
                    self.samples.append((path, object_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, object_name = self.samples[idx]
        pil_image = self.loader(path).convert("RGB") # Ensure image is RGB
        #print(path)
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        return pil_image, object_name


class COCOPositive(Dataset):
    """
    Creates a dataset of positive samples from COCO for a given list of object names.
    For an image containing multiple desired objects, it creates a separate sample for each.
    Returns a image and the corresponding object_name.
    """
    def __init__(self, root_dir, object_names, transform=None):
        """
        Args:
            coco_dir (string): Root directory of the COCO dataset.
            object_names (list): A list of object names to include as positive samples.
            split (string): The dataset split, e.g., 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.loader = default_loader
        self.samples = []
        
        # Get all class names from the subdirectories and sort them
        self.object_names = ["traffic light", "carrot", "toilet","knife","bottle","vase","clock","bus","boat","suitcase"]

        for object_name in self.object_names:
            object_dir = self.root_dir.format(cls=object_name)
            for img_file in sorted(os.listdir(object_dir)):
                # Ensure we are only picking up image files
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    path = os.path.join(object_dir, img_file)
                    self.samples.append((path, object_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, object_name = self.samples[idx]
        pil_image = self.loader(path).convert("RGB") # Ensure image is RGB
        #print(path)
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        return pil_image, object_name


class ObjectNet(Dataset):
    """
    ObjectNet dataset loader. ObjectNet is organized with folders for each object class.
    Each folder contains images of that object class.
    """
    def __init__(self, objectnet_dir, transform=None):
        """
        Args:
            objectnet_dir (string): Root directory of the ObjectNet dataset (should contain 'images' and 'mappings' folders)
            transform (tuple or callable, optional): Optional transform. If tuple (H, W), resizes images.
        """
        self.image_dir = os.path.join(objectnet_dir, "images")
        self.transform = transform
        self.loader = default_loader
        
        # Load mapping from folder names to labels
        mapping_path = os.path.join(objectnet_dir, "mappings", "folder_to_objectnet_label.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                self.folder_to_label = json.load(f)
        else:
            self.folder_to_label = {}
        
        # Build image index: (image_id, path, object_label)
        self.samples = []
        self.img_id_to_info = {}
        self.label_to_folders = defaultdict(list)
        
        # Get all object folders
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        object_folders = [d for d in os.listdir(self.image_dir) 
                         if os.path.isdir(os.path.join(self.image_dir, d))]
        
        img_id_counter = 0
        for folder_name in sorted(object_folders):
            folder_path = os.path.join(self.image_dir, folder_name)
            # Get label from mapping, or use folder name as fallback
            label = self.folder_to_label.get(folder_name, folder_name.replace('_', ' ').title())
            self.label_to_folders[label].append(folder_name)
            
            # Get all images in this folder
            for img_file in sorted(os.listdir(folder_path)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(folder_path, img_file)
                    img_id = img_id_counter
                    img_id_counter += 1
                    
                    self.samples.append((img_id, img_path, label))
                    self.img_id_to_info[img_id] = {
                        'path': img_path,
                        'label': label,
                        'folder': folder_name
                    }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Check if idx is a valid image ID (from get_imgIds)
        if idx in self.img_id_to_info:
            img_info = self.img_id_to_info[idx]
            img_path = img_info['path']
        else:
            # Otherwise, treat as index into samples list
            img_id, img_path, label = self.samples[idx]
        
        image = self.loader(img_path).convert("RGB")
        if self.transform is not None:
            if isinstance(self.transform, tuple):
                # Resize to (H, W)
                image = image.resize(self.transform, Image.LANCZOS)
            else:
                image = self.transform(image)
        
        return image, img_path
    
    def get_targets(self, idx):
        """Return list of object labels for an image (ObjectNet images have single label)"""
        if idx in self.img_id_to_info:
            label = self.img_id_to_info[idx]['label']
        else:
            _, _, label = self.samples[idx]
        return [label]
    
    def get_imgIds(self):
        """Return list of all image IDs"""
        return list(self.img_id_to_info.keys())
    
    def get_all_targets_names(self):
        """Return list of all unique object labels"""
        return list(set(self.label_to_folders.keys()))
    
    def get_categories(self, category=None):
        """For compatibility with COCO interface. Returns all categories or specific category."""
        all_cats = self.get_all_targets_names()
        if category is None:
            return all_cats
        # ObjectNet doesn't have supercategories, so return empty list or all if category matches
        return all_cats
    
    def get_all_supercategories(self):
        """ObjectNet doesn't have supercategories, return empty set for compatibility"""
        return set()
    
    def get_imgIds_by_class(self, present_classes=[], absent_classes=[]):
        """
        Return image IDs that have at least one of present_classes and none of absent_classes.
        For ObjectNet, we check the folder name (normalized) against the target object name.
        """
        ids = []
        # Normalize class names for comparison (handle case and underscores)
        present_normalized = [c.lower().replace(' ', '_') for c in present_classes]
        absent_normalized = [c.lower().replace(' ', '_') for c in absent_classes]
        
        for img_id in self.get_imgIds():
            img_info = self.img_id_to_info[img_id]
            folder_name = img_info['folder'].lower()
            label = img_info['label'].lower()
            
            # Check if image has any present class
            has_present = False
            if not present_classes:  # If no present_classes specified, include all
                has_present = True
            else:
                for pc in present_normalized:
                    if pc in folder_name or pc in label:
                        has_present = True
                        break
            
            # Check if image has any absent class
            has_absent = False
            for ac in absent_normalized:
                if ac in folder_name or ac in label:
                    has_absent = True
                    break
            
            # Include if has present and doesn't have absent
            if has_present and not has_absent:
                ids.append(img_id)
        
        return ids
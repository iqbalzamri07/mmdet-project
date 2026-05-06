"""
Custom Multi-Label Video Dataset for MMAction2
"""
import json
import numpy as np
from mmaction.datasets import VideoDataset
from mmengine.registry import DATASETS

@DATASETS.register_module()
class MultiLabelVideoDataset(VideoDataset):
    """Video dataset that supports multi-label classification"""
    
    def load_data_list(self):
        """Load annotation file with multi-label support"""
        data_list = []
        ann_file = self.ann_file
        
        # Check if JSON format (multi-label)
        if ann_file.endswith('.json'):
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                data_info = dict(
                    filename=ann['filename'],
                    labels=np.array(ann['labels'], dtype=np.float32),  # Multi-hot vector
                    primary_label=ann['primary_label'],
                )
                data_list.append(data_info)
        else:
            # Fallback to single-label txt format
            super().load_data_list()
            return self.data_list
        
        return data_list
    
    def parse_data_info(self, raw_data_info):
        """Parse raw data info to standard format"""
        data_info = {}
        data_info['filename'] = raw_data_info['filename']
        data_info['labels'] = raw_data_info['labels']  # Keep as multi-hot
        data_info['primary_label'] = raw_data_info.get('primary_label', 0)
        return data_info
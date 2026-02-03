import torch
import random
from typing import List, Dict
import json
import open_clip

class EnhancedTextRepresentations:
    
    def __init__(self, clip_model, tokenizer, coco_captions_path: str = None):
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.coco_captions = self._load_coco_captions(coco_captions_path) if coco_captions_path else None
        
    def _load_coco_captions(self, captions_path: str) -> Dict:
        """Load COCO captions for mining contextual descriptions"""
        with open(captions_path, 'r') as f:
            return json.load(f)
    
    def get_contextual_templates(self, obj: str) -> List[str]:
 
        
        generic_templates = [
            "A scene featuring a {obj}",
            "An image showing a {obj}",
            "A photograph with a {obj}",
            "A picture containing a {obj}"
        ]
        
        return  generic_templates
    
    def mine_captions_with_object(self, obj: str, max_captions: int = 50) -> List[str]:
        """Mine COCO captions that contain the target object"""
        if not self.coco_captions:
            return []
            
        matching_captions = []
        for annotation in self.coco_captions['annotations']:
            caption = annotation['caption'].lower()
            if obj.lower() in caption and len(matching_captions) < max_captions:
                matching_captions.append(annotation['caption'])
        
        return matching_captions
    
    def get_compositional_embeddings(self, obj: str, device: str = 'cuda') -> torch.Tensor:
        """Create compositional text embeddings combining multiple representations"""
        
        # 1. Simple sentence embedding (better than raw token)
        simple_sentence = f"A photo of a {obj}"
        simple_tokens = self.tokenizer([simple_sentence]).to(device)
        simple_embed = self.clip_model.encode_text(simple_tokens)
        
        # 2. Contextual template embeddings
        contextual_templates = self.get_contextual_templates(obj)
        context_embeds = []
        for template in contextual_templates[:3]:  # Use top 3 templates
            text = template.format(obj=obj)
            tokens = self.tokenizer([text]).to(device)
            embed = self.clip_model.encode_text(tokens)
            context_embeds.append(embed)
        
        # 3. COCO caption embeddings (if available)
        caption_embeds = []
        if self.coco_captions:
            captions = self.mine_captions_with_object(obj, max_captions=5)
            for caption in captions:
                tokens = self.tokenizer([caption]).to(device)
                embed = self.clip_model.encode_text(tokens)
                caption_embeds.append(embed)
        
        # 4. Combine embeddings with weighted average
        all_embeds = [simple_embed] + context_embeds + caption_embeds
        weights = torch.tensor([0.3] + [0.4/len(context_embeds)]*len(context_embeds) + 
                              [0.3/len(caption_embeds)]*len(caption_embeds)).to(device)
        
        if len(all_embeds) > 1:
            combined_embed = torch.stack(all_embeds).squeeze(1)
            weighted_embed = torch.sum(combined_embed * weights.unsqueeze(-1), dim=0)
            return weighted_embed
        else:
            return simple_embed.squeeze(0)

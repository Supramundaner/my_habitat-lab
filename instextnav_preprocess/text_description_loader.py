"""
Text Description Loader for TextNav
Loads and extracts text descriptions from val_text.json for specific objects.
"""

import json
import os
from typing import Dict, Any, Optional, Tuple

class TextDescriptionLoader:
    """Loads text descriptions for objects from val_text.json"""
    
    def __init__(self, val_text_path: str):
        """
        Initialize the text description loader.
        
        Args:
            val_text_path: Path to val_text.json file
        """
        self.val_text_path = val_text_path
        self.attribute_data = self._load_attribute_data()
    
    def _load_attribute_data(self) -> Dict[str, Any]:
        """Load attribute data from val_text.json"""
        if not os.path.exists(self.val_text_path):
            raise FileNotFoundError(f"val_text.json not found at: {self.val_text_path}")
        
        try:
            with open(self.val_text_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'attribute_data' not in data:
                raise KeyError("'attribute_data' key not found in val_text.json")
            
            print(f"✓ Loaded attribute data for {len(data['attribute_data'])} objects")
            return data['attribute_data']
            
        except Exception as e:
            raise RuntimeError(f"Failed to load val_text.json: {e}")
    
    def get_object_description(self, scene_id: str, object_id: str) -> Optional[Tuple[str, str]]:
        """
        Get text description for a specific object.
        
        Args:
            scene_id: Scene identifier (e.g., "4ok3usBNeis")
            object_id: Object ID (e.g., "8")
            
        Returns:
            Tuple of (intrinsic_attributes, extrinsic_attributes) or None if not found
        """
        # Construct the key in the format used in val_text.json
        object_key = f"{scene_id}_{object_id}"
        
        if object_key not in self.attribute_data:
            print(f"⚠️ Object key '{object_key}' not found in attribute data")
            return None
        
        obj_data = self.attribute_data[object_key]
        
        intrinsic = obj_data.get('intrinsic_attributes', '')
        extrinsic = obj_data.get('extrinsic_attributes', '')
        
        print(f"✓ Found text description for {object_key}")
        print(f"  - Intrinsic: {intrinsic[:100]}...")
        print(f"  - Extrinsic: {extrinsic[:100]}...")
        
        return (intrinsic, extrinsic)
    
    def create_combined_description(self, scene_id: str, object_id: str, 
                                  object_category: str) -> Optional[str]:
        """
        Create a combined text description for the object.
        
        Args:
            scene_id: Scene identifier
            object_id: Object ID
            object_category: Object category (e.g., "chair")
            
        Returns:
            Combined description string or None if not found
        """
        description_tuple = self.get_object_description(scene_id, object_id)
        
        if description_tuple is None:
            return None
        
        intrinsic, extrinsic = description_tuple
        
        # Create a combined description
        combined_description = f"""Target Object: {object_category} (ID: {object_id})

Intrinsic Attributes:
{intrinsic}

Extrinsic Attributes:
{extrinsic}"""
        
        return combined_description
    
    def extract_scene_id_from_path(self, scene_path: str) -> str:
        """
        Extract scene ID from scene path.
        
        Args:
            scene_path: Path to scene file (e.g., "/path/to/4ok3usBNeis.basis.glb")
            
        Returns:
            Scene ID (e.g., "4ok3usBNeis")
        """
        # Extract filename from path
        filename = os.path.basename(scene_path)
        
        # Remove .basis.glb extension
        if filename.endswith('.basis.glb'):
            scene_id = filename.replace('.basis.glb', '')
        elif filename.endswith('.glb'):
            scene_id = filename.replace('.glb', '')
        else:
            # Try to extract from directory structure
            # e.g., "/path/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
            parts = scene_path.split('/')
            for part in parts:
                if '-' in part and len(part.split('-')) == 2:
                    scene_id = part.split('-')[1]
                    break
            else:
                scene_id = filename
        
        print(f"📁 Extracted scene ID: '{scene_id}' from path: {scene_path}")
        return scene_id
    
    def validate_object_exists(self, scene_id: str, object_id: str) -> bool:
        """
        Check if an object exists in the attribute data.
        
        Args:
            scene_id: Scene identifier
            object_id: Object ID
            
        Returns:
            True if object exists, False otherwise
        """
        object_key = f"{scene_id}_{object_id}"
        exists = object_key in self.attribute_data
        
        if not exists:
            print(f"⚠️ Object {object_key} not found in attribute data")
            available_keys = [k for k in self.attribute_data.keys() if k.startswith(scene_id)]
            if available_keys:
                print(f"   Available objects for scene {scene_id}: {available_keys[:5]}...")
            else:
                print(f"   No objects found for scene {scene_id}")
        
        return exists


def load_text_description(val_text_path: str, scene_path: str, object_id: str, 
                         object_category: str) -> Optional[str]:
    """
    Convenience function to load text description for an object.
    
    Args:
        val_text_path: Path to val_text.json
        scene_path: Path to scene file
        object_id: Object ID
        object_category: Object category
        
    Returns:
        Combined text description or None if not found
    """
    try:
        loader = TextDescriptionLoader(val_text_path)
        scene_id = loader.extract_scene_id_from_path(scene_path)
        return loader.create_combined_description(scene_id, object_id, object_category)
    except Exception as e:
        print(f"✗ Failed to load text description: {e}")
        return None

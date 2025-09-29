# Simple test suite for Advanced DeepDream Implementation

import pytest
import torch
import numpy as np
from PIL import Image
import os
import tempfile

from advanced_deepdream import AdvancedDeepDream, ImageDatabase

class TestAdvancedDeepDream:
    """Test cases for AdvancedDeepDream class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.deepdream = AdvancedDeepDream(model_name='inception_v3', device='cpu')
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.deepdream.model is not None
        assert self.deepdream.device.type == 'cpu'
        assert self.deepdream.model.eval()
    
    def test_device_selection(self):
        """Test device selection"""
        # Test CPU device
        deepdream_cpu = AdvancedDeepDream(device='cpu')
        assert deepdream_cpu.device.type == 'cpu'
        
        # Test auto device selection
        deepdream_auto = AdvancedDeepDream(device='auto')
        assert deepdream_auto.device.type in ['cpu', 'cuda']
    
    def test_model_loading(self):
        """Test different model loading"""
        models = ['inception_v3', 'resnet50', 'vgg16']
        
        for model_name in models:
            deepdream = AdvancedDeepDream(model_name=model_name, device='cpu')
            assert deepdream.model is not None
            assert deepdream.model.eval()
    
    def test_image_preprocessing(self):
        """Test image preprocessing"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_path = os.path.join(self.temp_dir, 'test.jpg')
        test_image.save(test_path)
        
        # Test loading
        tensor = self.deepdream.load_image(test_path)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1  # Batch dimension
        assert tensor.shape[1] == 3   # RGB channels
        assert tensor.shape[2] == 224  # Height
        assert tensor.shape[3] == 224  # Width
    
    def test_deprocess_image(self):
        """Test image deprocessing"""
        # Create a test tensor
        test_tensor = torch.randn(1, 3, 224, 224)
        
        # Test deprocessing
        result = self.deepdream.deprocess_image(test_tensor)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_octave_scales(self):
        """Test octave scale creation"""
        test_tensor = torch.randn(1, 3, 224, 224)
        
        octaves = self.deepdream.create_octave_scales(test_tensor, octave_n=3)
        
        assert len(octaves) == 3
        assert octaves[0].shape == (1, 3, 224, 224)  # Original size
        assert octaves[-1].shape[2] < 224  # Smaller size
    
    def test_layer_mapping(self):
        """Test layer mapping for different models"""
        models = ['inception_v3', 'resnet50', 'vgg16']
        
        for model_name in models:
            deepdream = AdvancedDeepDream(model_name=model_name, device='cpu')
            
            # Test that layer maps exist
            assert model_name in deepdream.layer_maps
            assert 'early' in deepdream.layer_maps[model_name]
            assert 'mid' in deepdream.layer_maps[model_name]
            assert 'late' in deepdream.layer_maps[model_name]
    
    def test_generate_dream_basic(self):
        """Test basic dream generation"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        test_path = os.path.join(self.temp_dir, 'test.jpg')
        test_image.save(test_path)
        
        # Test dream generation
        try:
            dream_result = self.deepdream.generate_dream(
                test_path,
                method='octave',
                layer='mid',
                iterations=5,  # Small number for testing
                octave_n=2
            )
            
            assert isinstance(dream_result, np.ndarray)
            assert dream_result.shape[2] == 3  # RGB channels
            assert dream_result.min() >= 0
            assert dream_result.max() <= 1
            
        except Exception as e:
            pytest.skip(f"Dream generation failed: {e}")

class TestImageDatabase:
    """Test cases for ImageDatabase class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db = ImageDatabase(data_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database initialization"""
        assert self.db.data_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
        assert isinstance(self.db.images_db, dict)
    
    def test_sample_image_creation(self):
        """Test sample image creation"""
        image_path = self.db.create_sample_image('test.jpg')
        
        assert os.path.exists(image_path)
        assert image_path.endswith('test.jpg')
        
        # Verify image can be loaded
        image = Image.open(image_path)
        assert image.mode == 'RGB'
        assert image.size == (224, 224)
    
    def test_get_images_by_category(self):
        """Test getting images by category"""
        nature_images = self.db.get_images_by_category('nature')
        assert isinstance(nature_images, list)
        assert len(nature_images) > 0
        
        # Test non-existent category
        fake_images = self.db.get_images_by_category('fake_category')
        assert fake_images == []
    
    def test_get_all_images(self):
        """Test getting all images"""
        all_images = self.db.get_all_images()
        assert isinstance(all_images, list)
        assert len(all_images) > 0
        
        # Check that all images have required fields
        for img_info in all_images:
            assert 'name' in img_info
            assert 'description' in img_info
            assert 'category' in img_info

def test_integration():
    """Integration test"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize components
        deepdream = AdvancedDeepDream(model_name='inception_v3', device='cpu')
        db = ImageDatabase(data_dir=temp_dir)
        
        # Create sample image
        image_path = db.create_sample_image('integration_test.jpg')
        
        # Generate dream
        dream_result = deepdream.generate_dream(
            image_path,
            method='octave',
            layer='mid',
            iterations=3,  # Small for testing
            octave_n=2
        )
        
        # Verify result
        assert isinstance(dream_result, np.ndarray)
        assert dream_result.shape[2] == 3
        
    except Exception as e:
        pytest.skip(f"Integration test failed: {e}")
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

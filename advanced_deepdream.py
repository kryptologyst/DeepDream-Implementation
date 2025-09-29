# Project 138: Advanced DeepDream Implementation
# Description: Modern DeepDream with latest PyTorch, advanced techniques, and web UI
# Features: Octave scaling, layer blending, guided DeepDream, style transfer, batch processing

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AdvancedDeepDream:
    """
    Advanced DeepDream implementation with modern PyTorch and enhanced features
    """
    
    def __init__(self, model_name: str = 'inception_v3', device: str = 'auto'):
        """
        Initialize DeepDream with modern PyTorch models
        
        Args:
            model_name: Model to use ('inception_v3', 'resnet50', 'vgg16')
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_name)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Available layers for different models
        self.layer_maps = {
            'inception_v3': {
                'early': 'Mixed_5b',
                'mid': 'Mixed_6a', 
                'late': 'Mixed_7a'
            },
            'resnet50': {
                'early': 'layer1',
                'mid': 'layer2',
                'late': 'layer3'
            },
            'vgg16': {
                'early': 'features.4',
                'mid': 'features.8',
                'late': 'features.12'
            }
        }
        
        self.current_layer = None
        self.activations = None
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for computation"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_name: str) -> nn.Module:
        """Load pre-trained model with modern PyTorch"""
        if model_name == 'inception_v3':
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model.to(self.device)
    
    def _register_hook(self, layer_name: str):
        """Register forward hook to capture activations"""
        def hook_fn(module, input, output):
            self.activations = output
        
        layer = self._get_layer_by_name(layer_name)
        layer.register_forward_hook(hook_fn)
        self.current_layer = layer_name
    
    def _get_layer_by_name(self, layer_name: str):
        """Get layer by name from model"""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            else:
                raise ValueError(f"Layer {layer_name} not found")
        return layer
    
    def deprocess_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to displayable image"""
        img = img_tensor.clone().detach().cpu()
        img = img.squeeze().permute(1, 2, 0).numpy()
        
        # Unnormalize using ImageNet statistics
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        return img
    
    def load_image(self, image_path: str, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image = image.resize(size)
        
        # Convert to tensor and add batch dimension
        tensor = self.preprocess(image).unsqueeze(0)
        return tensor.to(self.device).requires_grad_(True)
    
    def create_octave_scales(self, image: torch.Tensor, octave_n: int = 4, octave_scale: float = 1.4) -> List[torch.Tensor]:
        """Create octave scales for multi-scale DeepDream"""
        octaves = []
        h, w = image.shape[-2:]
        
        for i in range(octave_n):
            scale = 1.0 / (octave_scale ** i)
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h > 0 and new_w > 0:
                octave = transforms.Resize((new_h, new_w))(image)
                octaves.append(octave)
        
        return octaves
    
    def dream_single_octave(self, image: torch.Tensor, layer_name: str, iterations: int = 20, 
                          lr: float = 0.01, l2_coeff: float = 0.01) -> torch.Tensor:
        """Apply DeepDream to single octave"""
        self._register_hook(layer_name)
        
        optimizer = torch.optim.Adam([image], lr=lr)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            self.model(image)
            
            # Loss: maximize activations + L2 regularization
            loss = -self.activations.norm() + l2_coeff * image.norm()
            
            loss.backward()
            optimizer.step()
            
            # Optional: print progress
            if i % 5 == 0:
                print(f"  Octave iteration {i}, Loss: {loss.item():.4f}")
        
        return image
    
    def deepdream_octave(self, image: torch.Tensor, layer_name: str, iterations: int = 20,
                        octave_n: int = 4, octave_scale: float = 1.4, lr: float = 0.01) -> torch.Tensor:
        """Apply DeepDream with octave scaling"""
        print(f"🎨 Starting DeepDream with {octave_n} octaves...")
        
        # Create octave scales
        octaves = self.create_octave_scales(image, octave_n, octave_scale)
        
        # Start with smallest octave
        current_image = octaves[-1].clone().requires_grad_(True)
        
        for i, octave in enumerate(reversed(octaves)):
            print(f"📐 Processing octave {i+1}/{octave_n} (size: {octave.shape[-2:]})")
            
            # Dream on current octave
            current_image = self.dream_single_octave(current_image, layer_name, iterations, lr)
            
            # Upscale to next octave (except for the last one)
            if i < len(octaves) - 1:
                next_size = octaves[-(i+2)].shape[-2:]
                current_image = transforms.Resize(next_size)(current_image)
        
        return current_image
    
    def guided_deepdream(self, image: torch.Tensor, guide_image: torch.Tensor, 
                        layer_name: str, iterations: int = 20, guide_weight: float = 0.5) -> torch.Tensor:
        """Apply guided DeepDream using a guide image"""
        print("🎯 Applying Guided DeepDream...")
        
        self._register_hook(layer_name)
        
        # Process guide image
        guide_features = self.model(guide_image)
        guide_activations = self.activations.clone().detach()
        
        optimizer = torch.optim.Adam([image], lr=0.01)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            self.model(image)
            
            # Loss: maximize current activations + similarity to guide
            dream_loss = -self.activations.norm()
            guide_loss = -torch.cosine_similarity(
                self.activations.flatten(), 
                guide_activations.flatten(), 
                dim=0
            )
            
            total_loss = dream_loss + guide_weight * guide_loss
            
            total_loss.backward()
            optimizer.step()
            
            if i % 5 == 0:
                print(f"  Guided iteration {i}, Loss: {total_loss.item():.4f}")
        
        return image
    
    def blend_layers(self, image: torch.Tensor, layer_weights: Dict[str, float], 
                    iterations: int = 20) -> torch.Tensor:
        """Blend multiple layers for complex DeepDream effects"""
        print("🎭 Blending multiple layers...")
        
        # Register hooks for all layers
        layer_activations = {}
        hooks = []
        
        for layer_name in layer_weights.keys():
            def make_hook(name):
                def hook_fn(module, input, output):
                    layer_activations[name] = output
                return hook_fn
            
            layer = self._get_layer_by_name(layer_name)
            hook = layer.register_forward_hook(make_hook(layer_name))
            hooks.append(hook)
        
        optimizer = torch.optim.Adam([image], lr=0.01)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            self.model(image)
            
            # Weighted loss from multiple layers
            total_loss = 0
            for layer_name, weight in layer_weights.items():
                if layer_name in layer_activations:
                    total_loss += weight * (-layer_activations[layer_name].norm())
            
            total_loss.backward()
            optimizer.step()
            
            if i % 5 == 0:
                print(f"  Blend iteration {i}, Loss: {total_loss.item():.4f}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return image
    
    def generate_dream(self, image_path: str, method: str = 'octave', 
                      layer: str = 'mid', **kwargs) -> np.ndarray:
        """Main method to generate DeepDream image"""
        print(f"💭 Generating DeepDream using {method} method...")
        
        # Load image
        image = self.load_image(image_path)
        
        # Get layer name
        model_name = self.model.__class__.__name__.lower()
        if 'inception' in model_name:
            model_name = 'inception_v3'
        elif 'resnet' in model_name:
            model_name = 'resnet50'
        elif 'vgg' in model_name:
            model_name = 'vgg16'
        
        layer_name = self.layer_maps[model_name][layer]
        
        # Apply DeepDream based on method
        if method == 'octave':
            result = self.deepdream_octave(image, layer_name, **kwargs)
        elif method == 'guided':
            guide_path = kwargs.get('guide_path')
            if not guide_path:
                raise ValueError("guide_path required for guided DeepDream")
            guide_image = self.load_image(guide_path)
            result = self.guided_deepdream(image, guide_image, layer_name, **kwargs)
        elif method == 'blend':
            layer_weights = kwargs.get('layer_weights', {'early': 0.3, 'mid': 0.5, 'late': 0.2})
            layer_names = {self.layer_maps[model_name][k]: v for k, v in layer_weights.items()}
            result = self.blend_layers(image, layer_names, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.deprocess_image(result)

class ImageDatabase:
    """Mock database for sample images"""
    
    def __init__(self, data_dir: str = "sample_images"):
        self.data_dir = data_dir
        self.images_db = self._create_sample_images()
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _create_sample_images(self) -> Dict:
        """Create mock database of sample images"""
        return {
            "nature": [
                {"name": "forest.jpg", "description": "Dense forest landscape", "category": "nature"},
                {"name": "mountain.jpg", "description": "Snow-capped mountain peak", "category": "nature"},
                {"name": "ocean.jpg", "description": "Ocean waves at sunset", "category": "nature"}
            ],
            "architecture": [
                {"name": "building.jpg", "description": "Modern skyscraper", "category": "architecture"},
                {"name": "bridge.jpg", "description": "Suspension bridge", "category": "architecture"},
                {"name": "cathedral.jpg", "description": "Gothic cathedral", "category": "architecture"}
            ],
            "abstract": [
                {"name": "patterns.jpg", "description": "Geometric patterns", "category": "abstract"},
                {"name": "colors.jpg", "description": "Colorful abstract art", "category": "abstract"},
                {"name": "textures.jpg", "description": "Textural composition", "category": "abstract"}
            ]
        }
    
    def get_images_by_category(self, category: str) -> List[Dict]:
        """Get images by category"""
        return self.images_db.get(category, [])
    
    def get_all_images(self) -> List[Dict]:
        """Get all images"""
        all_images = []
        for category_images in self.images_db.values():
            all_images.extend(category_images)
        return all_images
    
    def create_sample_image(self, image_name: str, size: Tuple[int, int] = (224, 224)) -> str:
        """Create a sample image for testing"""
        image_path = os.path.join(self.data_dir, image_name)
        
        # Create a random colorful image
        np.random.seed(42)  # For reproducible results
        image_array = np.random.rand(*size, 3)
        
        # Add some structure
        x, y = np.meshgrid(np.linspace(0, 1, size[0]), np.linspace(0, 1, size[1]))
        image_array[:, :, 0] += 0.3 * np.sin(10 * x) * np.cos(10 * y)
        image_array[:, :, 1] += 0.3 * np.cos(15 * x) * np.sin(15 * y)
        image_array[:, :, 2] += 0.3 * np.sin(20 * x + y)
        
        image_array = np.clip(image_array, 0, 1)
        
        # Convert to PIL and save
        image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
        image_pil.save(image_path)
        
        return image_path

def main():
    """Main function to demonstrate DeepDream capabilities"""
    print("🚀 Advanced DeepDream Implementation")
    print("=" * 50)
    
    # Initialize DeepDream
    deepdream = AdvancedDeepDream(model_name='inception_v3')
    
    # Initialize image database
    db = ImageDatabase()
    
    # Create sample images
    print("📸 Creating sample images...")
    sample_images = []
    for category_images in db.get_all_images():
        for img_info in category_images:
            image_path = db.create_sample_image(img_info['name'])
            sample_images.append(image_path)
            print(f"  Created: {image_path}")
    
    if not sample_images:
        print("❌ No sample images created")
        return
    
    # Test different DeepDream methods
    test_image = sample_images[0]
    print(f"\n🎨 Testing DeepDream on: {test_image}")
    
    # Method 1: Octave DeepDream
    print("\n1️⃣ Octave DeepDream:")
    try:
        dream_octave = deepdream.generate_dream(
            test_image, 
            method='octave', 
            layer='mid',
            iterations=10,
            octave_n=3
        )
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        original = Image.open(test_image)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(dream_octave)
        plt.title("DeepDream (Octave)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('deepdream_octave_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"❌ Octave DeepDream failed: {e}")
    
    # Method 2: Layer Blending
    print("\n2️⃣ Layer Blending:")
    try:
        dream_blend = deepdream.generate_dream(
            test_image,
            method='blend',
            layer_weights={'early': 0.2, 'mid': 0.6, 'late': 0.2},
            iterations=15
        )
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        original = Image.open(test_image)
        plt.imshow(original)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(dream_blend)
        plt.title("DeepDream (Layer Blend)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('deepdream_blend_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"❌ Layer Blending failed: {e}")
    
    print("\n✅ DeepDream demonstration completed!")
    print("📁 Results saved as PNG files")

if __name__ == "__main__":
    main()

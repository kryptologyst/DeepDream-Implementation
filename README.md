# DeepDream Implementation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern, advanced implementation of DeepDream using state-of-the-art PyTorch models with interactive web interface, multiple neural networks, and cutting-edge techniques.

## Features

### **Advanced DeepDream Techniques**
- **Octave Scaling**: Multi-scale processing for superior detail preservation
- **Layer Blending**: Combine multiple neural layers for complex artistic effects
- **Guided DeepDream**: Use reference images to guide the dream's style
- **Multiple Models**: InceptionV3, ResNet50, and VGG16 support
- **Layer Selection**: Early, mid, and late layer targeting for different effects

### **Interactive Web Interface**
- **Modern Streamlit UI**: Beautiful, responsive web interface
- **Real-time Generation**: Watch your dreams come to life
- **Parameter Tuning**: Adjust all settings with intuitive controls
- **Sample Gallery**: Built-in sample images for testing
- **Download Results**: Save your creations in high quality

### **Technical Excellence**
- **Modern PyTorch**: Latest APIs and best practices
- **GPU Acceleration**: Automatic CUDA support when available
- **Production Ready**: Robust error handling and optimization
- **Extensible Architecture**: Easy to add new models and techniques

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kryptologyst/DeepDream-Implementation.git
cd DeepDream-Implementation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the web interface**
```bash
streamlit run streamlit_app.py
```

4. **Or run the command-line version**
```bash
python advanced_deepdream.py
```

### Basic Usage

```python
from advanced_deepdream import AdvancedDeepDream

# Initialize DeepDream
deepdream = AdvancedDeepDream(model_name='inception_v3')

# Generate a dream
dream_image = deepdream.generate_dream(
    'your_image.jpg',
    method='octave',
    layer='mid',
    iterations=20
)
```

## Documentation

### DeepDream Methods

#### **Octave Scaling**
Processes images at multiple scales for better detail preservation:
```python
dream_image = deepdream.generate_dream(
    'image.jpg',
    method='octave',
    octave_n=4,        # Number of octaves
    octave_scale=1.4,  # Scale factor between octaves
    iterations=20
)
```

#### **Layer Blending**
Combines multiple neural layers for complex effects:
```python
dream_image = deepdream.generate_dream(
    'image.jpg',
    method='blend',
    layer_weights={
        'early': 0.2,  # Simple patterns
        'mid': 0.6,    # Complex shapes
        'late': 0.2    # Abstract concepts
    },
    iterations=20
)
```

#### **Guided DeepDream**
Uses a reference image to guide the style:
```python
dream_image = deepdream.generate_dream(
    'image.jpg',
    method='guided',
    guide_path='style_reference.jpg',
    guide_weight=0.5,
    iterations=20
)
```

### Available Models

| Model | Best For | Characteristics |
|-------|----------|----------------|
| **InceptionV3** | Nature, organic shapes | Flowing, organic patterns |
| **ResNet50** | Architecture, objects | Balanced detail and abstraction |
| **VGG16** | Textures, patterns | Rich textures and fine details |

### Layer Selection

| Layer | Captures | Visual Effect |
|-------|----------|---------------|
| **Early** | Edges, textures, simple patterns | Subtle enhancements |
| **Mid** | Objects, shapes, structures | Moderate surrealism |
| **Late** | Abstract concepts, high-level features | Highly abstract, dream-like |

## Web Interface Guide

### **Main Features**

1. **Image Upload**: Drag and drop or click to upload your images
2. **Model Selection**: Choose between InceptionV3, ResNet50, or VGG16
3. **Method Selection**: Pick from Octave Scaling, Layer Blending, or Guided DeepDream
4. **Parameter Tuning**: Adjust iterations, learning rate, and method-specific settings
5. **Real-time Preview**: See your results instantly
6. **Download**: Save your creations in PNG format

### **Sample Gallery**
Built-in sample images organized by category:
- **Nature**: Forest, mountain, ocean scenes
- **Architecture**: Buildings, bridges, structures
- **Abstract**: Patterns, colors, textures

## 🔧 Advanced Configuration

### **Custom Parameters**

```python
# Advanced configuration
deepdream = AdvancedDeepDream(
    model_name='inception_v3',
    device='cuda'  # or 'cpu', 'auto'
)

# Custom preprocessing
dream_image = deepdream.generate_dream(
    'image.jpg',
    method='octave',
    layer='mid',
    iterations=30,
    lr=0.01,
    octave_n=5,
    octave_scale=1.3
)
```

### **Batch Processing**

```python
# Process multiple images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = []

for path in image_paths:
    dream = deepdream.generate_dream(path, method='octave')
    results.append(dream)
```

## 📁 Project Structure

```
advanced-deepdream/
├── advanced_deepdream.py    # Core DeepDream implementation
├── streamlit_app.py         # Web interface
├── 0138.py                  # Original implementation
├── requirements.txt         # Dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── sample_images/          # Generated sample images
└── results/                # Generated DeepDream images
```

## Testing

### **Run Tests**
```bash
python -m pytest tests/
```

### **Generate Sample Images**
```python
from advanced_deepdream import ImageDatabase

db = ImageDatabase()
sample_images = db.create_sample_image('test.jpg')
```

## Performance Tips

### **Optimization**
- Use GPU when available (`device='cuda'`)
- Reduce iterations for faster generation
- Lower learning rates for smoother results
- Use fewer octaves for speed

### **Quality Tips**
- Higher iterations = more detailed dreams
- Experiment with different models
- Try layer blending for unique effects
- Use guided DeepDream for specific styles

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
git clone https://github.com/yourusername/advanced-deepdream.git
cd advanced-deepdream
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### **Code Style**
- Follow PEP 8
- Use type hints
- Add docstrings
- Write tests for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google DeepDream Team**: Original DeepDream research
- **PyTorch Team**: Excellent deep learning framework
- **Streamlit Team**: Amazing web app framework
- **Computer Vision Community**: Continuous inspiration and research

## References

- [DeepDream: A Neural Network Visualization](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
- [Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Troubleshooting

### **Common Issues**

**CUDA Out of Memory**
```python
# Reduce batch size or use CPU
deepdream = AdvancedDeepDream(device='cpu')
```

**Slow Generation**
```python
# Reduce iterations and octaves
dream_image = deepdream.generate_dream(
    'image.jpg',
    iterations=10,
    octave_n=2
)
```

**Poor Quality Results**
```python
# Increase iterations and try different layers
dream_image = deepdream.generate_dream(
    'image.jpg',
    iterations=30,
    layer='late'
)
```

# DeepDream-Implementation

# Contributing to Advanced DeepDream Implementation

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of PyTorch and computer vision

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/advanced-deepdream.git
   cd advanced-deepdream
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📝 How to Contribute

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include Python version, OS, and error messages
- Provide minimal reproducible code
- Use the bug report template

### ✨ Feature Requests
- Check existing issues first
- Describe the feature clearly
- Explain the use case and benefits
- Use the feature request template

### 🔧 Code Contributions

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   python -m pytest tests/
   python -m pytest tests/ --cov=advanced_deepdream
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Coding Guidelines

### **Code Style**
- Follow PEP 8
- Use type hints for function parameters and return values
- Write descriptive docstrings for all functions and classes
- Use meaningful variable and function names

### **Example Code Style**
```python
def generate_dream(
    self, 
    image_path: str, 
    method: str = 'octave', 
    layer: str = 'mid',
    **kwargs
) -> np.ndarray:
    """
    Generate DeepDream image using specified method and parameters.
    
    Args:
        image_path: Path to input image
        method: DeepDream method ('octave', 'blend', 'guided')
        layer: Neural layer to target ('early', 'mid', 'late')
        **kwargs: Additional method-specific parameters
        
    Returns:
        Generated DeepDream image as numpy array
        
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If method or layer is invalid
    """
    # Implementation here
    pass
```

### **Testing**
- Write unit tests for new functions
- Test edge cases and error conditions
- Aim for >80% code coverage
- Use descriptive test names

### **Documentation**
- Update README.md for new features
- Add docstrings to all public functions
- Include usage examples
- Update type hints

## 🧪 Testing

### **Run Tests**
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=advanced_deepdream

# Run specific test file
python -m pytest tests/test_deepdream.py

# Run with verbose output
python -m pytest -v
```

### **Test Structure**
```
tests/
├── test_deepdream.py      # Core functionality tests
├── test_ui.py            # Streamlit UI tests
├── test_database.py      # Image database tests
└── fixtures/             # Test data and fixtures
```

## 📚 Documentation

### **Code Documentation**
- Use Google-style docstrings
- Include type hints
- Document all parameters and return values
- Add usage examples

### **README Updates**
- Update feature list for new capabilities
- Add installation instructions for new dependencies
- Include usage examples
- Update troubleshooting section

## 🎯 Areas for Contribution

### **High Priority**
- Additional neural network models (EfficientNet, Vision Transformer)
- Performance optimizations
- Better error handling
- More DeepDream techniques

### **Medium Priority**
- Batch processing improvements
- Video DeepDream support
- Advanced UI features
- Docker containerization

### **Low Priority**
- Additional sample images
- Documentation improvements
- Code refactoring
- Test coverage improvements

## 🔍 Review Process

### **Pull Request Guidelines**
- Keep PRs focused and small
- Include tests for new functionality
- Update documentation
- Ensure all tests pass
- Follow the PR template

### **Review Criteria**
- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Backward compatibility

## 🏷️ Release Process

### **Version Numbering**
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update version in setup.py and __init__.py
- Create release notes
- Tag releases in Git

### **Release Checklist**
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version numbers updated
- [ ] Release notes written
- [ ] Git tag created

## 💬 Communication

### **Getting Help**
- GitHub Discussions for questions
- GitHub Issues for bugs and features
- Email for security issues

### **Community Guidelines**
- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and experience
- Follow the code of conduct

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Advanced DeepDream Implementation! 🎨✨

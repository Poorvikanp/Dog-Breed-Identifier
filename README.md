# 🐕 Dog Breed Identifier

A production-ready machine learning system that can identify dog breeds from images with 78-80% accuracy. Built with TensorFlow, FastAPI, and deployed as a web application.

![Dog Breed Classification](https://img.shields.io/badge/ML-TensorFlow-orange) ![API](https://img.shields.io/badge/API-FastAPI-green) ![Deployment](https://img.shields.io/badge/Deployment-Docker-blue)

## 🎯 What This Project Does

Upload a photo of any dog and get an instant breed prediction with confidence score. Our system compares predictions from two different datasets to provide robust and reliable results.

**Demo**: [Try it live](https://your-deployed-app-url.com) *(deployment link)*

## ✨ Features

- **🤖 Dual-Dataset Analysis**: Compares predictions from Kaggle and Stanford datasets
- **⚡ Real-time Prediction**: ~52ms inference time
- **📱 User-Friendly Web Interface**: Clean, responsive design
- **🔄 Model Fallback**: Automatic fallback to ImageNet if custom models fail
- **📊 Performance Metrics**: Built-in accuracy tracking and reporting
- **🐳 Production Ready**: Docker containerized with health checks
- **📈 Research-Grade**: Academic-level documentation and analysis

## 🚀 Quick Start

### Option 1: Windows Quick Start
```bash
# Double-click or run:
run_server.bat
# Opens: http://127.0.0.1:9000/
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r ml_breed_classifier/requirements.txt

# 2. Run the server
uvicorn ml_breed_classifier.backend.app:app --reload --port 9000

# 3. Open browser
# http://localhost:9000/
```

### Option 3: Docker Deployment
```bash
# Build and run
docker build -t dog-breed-api -f ml_breed_classifier/Dockerfile .
docker run --rm -p 8000:8000 dog-breed-api
```

## 📊 Performance Results

| Dataset | Accuracy | Loss | Model Size | Inference Time |
|---------|----------|------|------------|----------------|
| **Kaggle** | 78.08% | 0.728 | 14MB | ~52ms |
| **Stanford** | 80.04% | 0.678 | 14MB | ~51ms |

**Key Insight**: Stanford Dogs Dataset (real-world photos) outperforms Kaggle (competition photos) by 2%, demonstrating the importance of data diversity over quantity.

## 🏗️ Project Architecture

```
📸 User Uploads Photo 
    ⬇️
🌐 FastAPI Web Service
    ⬇️
🧠 TensorFlow Model (MobileNetV2)
    ⬇️
📊 Prediction + Confidence Score
```

### Tech Stack
- **Backend**: FastAPI, Uvicorn, Starlette
- **ML Framework**: TensorFlow 2.20, Keras
- **Frontend**: HTML/CSS/JavaScript (vanilla)
- **Deployment**: Docker, Render.com ready
- **Data Processing**: NumPy, Pillow, pandas

## 📁 Project Structure

```
Dog-Breed-Identifier/
├── ml_breed_classifier/          # Main application
│   ├── backend/                  # FastAPI application
│   │   ├── app.py               # API endpoints
│   │   ├── registry.py          # Model management
│   │   └── static/              # Web interface
│   ├── scripts/                 # Training & data scripts
│   │   ├── train.py            # Model training
│   │   ├── download_data.py    # Data acquisition
│   │   └── prepare_dataset.py  # Data preprocessing
│   ├── requirements.txt         # Dependencies
│   ├── Dockerfile              # Containerization
│   └── README.md               # Detailed docs
├── models/                      # Trained models
│   ├── dataset1/               # Kaggle model
│   └── dataset2/               # Stanford model
├── PROJECT_OVERVIEW.md         # Technical overview
├── COMPREHENSIVE_PROJECT_GUIDE.md  # Complete documentation
└── CLASS_PRESENTATION_GUIDE.md     # Presentation materials
```

## 🔬 API Documentation

### Endpoints

**Single Prediction**
```bash
curl -F "file=@dog.jpg" "http://localhost:9000/predict?dataset=dataset1"
```

**Compare Both Datasets**
```bash
curl -F "file=@dog.jpg" "http://localhost:9000/predict_compare"
```

**Get Model Metrics**
```bash
curl "http://localhost:9000/metrics?dataset=dataset1"
```

**Health Check**
```bash
curl "http://localhost:9000/health"
# Returns: {"status": "ok"}
```

## 🧪 Training Your Own Models

### 1. Setup Kaggle API
```bash
# Place kaggle.json at %USERPROFILE%\.kaggle\kaggle.json
```

### 2. Download Datasets
```bash
python -m ml_breed_classifier.scripts.download_data --dataset both --out data
```

### 3. Prepare Data
```bash
python -m ml_breed_classifier.scripts.prepare_dataset --dataset both --inbase data --outbase data --val_ratio 0.15
```

### 4. Train Models
```bash
# Train Kaggle model
python -m ml_breed_classifier.scripts.train --dataset dataset1 --data_root data --models_root models --epochs 3

# Train Stanford model
python -m ml_breed_classifier.scripts.train --dataset dataset2 --data_root data --models_root models --epochs 3
```

## 🎓 Educational Resources

### For Students & Researchers
- **[Complete Project Guide](COMPREHENSIVE_PROJECT_GUIDE.md)**: 650+ line comprehensive documentation
- **[Class Presentation Guide](CLASS_PRESENTATION_GUIDE.md)**: Ready-to-present materials
- **[Technical Overview](PROJECT_OVERVIEW.md)**: Architecture and implementation details

### Key Learning Topics
- **Transfer Learning**: Using pre-trained models for new tasks
- **CNN Architectures**: MobileNetV2 efficiency vs accuracy trade-offs
- **Production ML**: From research to deployment
- **Model Interpretability**: Grad-CAM visualizations
- **API Design**: RESTful services with FastAPI

## 🔍 Research Context

This project builds upon research by [Cui et al. (2024)](https://bioengineeringjournal.org) who achieved 95.24% accuracy using ensemble methods. Our contribution:

- **Practical Deployment**: Production-ready web application
- **Efficiency Focus**: 14MB model vs ensemble approaches
- **Comparative Analysis**: Novel two-dataset methodology
- **Accessibility**: Works on any device with a web browser

## 🚀 Deployment Options

### Render.com (Recommended)
```yaml
# Uses ml_breed_classifier/render.yaml
# Auto-deploys from GitHub
```

### Cloud Platforms
- **Heroku**: Use provided Dockerfile
- **AWS/GCP**: Container deployment ready
- **Local**: Run with provided batch scripts

## 🛠️ Development

### Local Development Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r ml_breed_classifier/requirements.txt

# Run development server
uvicorn ml_breed_classifier.backend.app:app --reload
```

### Model Development
```bash
# Generate performance reports
python report_figures_generator_full.py

# Analyze model layers
python ml_breed_classifier/print_layers.py
```

## 📈 Performance Analysis

### Accuracy vs Efficiency Trade-offs
- **Our Approach**: 78-80% accuracy, 14MB model, 52ms inference
- **Research Papers**: 95% accuracy, large ensemble models, slower inference
- **Business Impact**: 7x size reduction with acceptable accuracy loss

### Why MobileNetV2?
- **Size**: 14MB vs 98MB (ResNet50)
- **Speed**: ~50ms vs ~80ms inference
- **Accuracy**: Good balance for production use
- **Deployment**: Mobile and edge-friendly

## 🎯 Real-World Applications

- **🏥 Veterinary Medicine**: Breed-specific health decisions
- **🏠 Animal Shelters**: Automated breed identification
- **💼 Pet Insurance**: Policy underwriting automation
- **🛍️ Pet Industry**: Personalized product recommendations

## 🔮 Future Roadmap

### Short Term (3-6 months)
- [ ] Mobile app development (TensorFlow Lite)
- [ ] Data augmentation implementation
- [ ] Batch processing capabilities
- [ ] User feedback system

### Medium Term (6-12 months)
- [ ] Vision Transformer integration
- [ ] Cross-species classification
- [ ] Real-time video analysis
- [ ] Advanced model interpretability

### Long Term (1-2 years)
- [ ] Multi-modal learning (text + images)
- [ ] Federated learning implementation
- [ ] Enterprise API development
- [ ] Integration with veterinary systems

## 📚 Documentation

- **[Technical Architecture](ml_breed_classifier/PROJECT_EXPLANATION.md)**: Deep-dive into system design
- **[Research Context](ml_breed_classifier/RESEARCH_PAPER_README.md)**: Academic background
- **[Validation Metrics](ml_breed_classifier/validation_metrics_explanation.md)**: Performance analysis
- **[Future Work](ml_breed_classifier/FUTURE_WORK_ANALYSIS.md)**: Research directions

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **Model Improvements**: Better architectures, training strategies
- **Mobile Apps**: iOS/Android implementations
- **Documentation**: Tutorials, examples, translations
- **Research**: Dataset analysis, comparative studies

### Getting Started
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Dataset Sources**: 
  - [Kaggle Dog Breed Identification](https://kaggle.com/c/dog-breed-identification)
  - [Stanford Dogs Dataset](https://kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
- **Research Foundation**: [Cui et al. (2024)](https://bioengineeringjournal.org) dog breed classification research
- **ML Framework**: TensorFlow/Keras community
- **Deployment**: Render.com for hosting

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Poorvikanp/Dog-Breed-Identifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Poorvikanp/Dog-Breed-Identifier/discussions)
- **Email**: poorvikanp@example.com

---

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ for the machine learning community*
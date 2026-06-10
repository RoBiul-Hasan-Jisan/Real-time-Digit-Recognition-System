#  Real-Time Handwritten Digit Recognition System

A full-stack Machine Learning web application that recognizes handwritten digits (0–9) in real time using a Convolutional Neural Network (CNN). Users can draw digits directly on an interactive canvas and instantly receive predictions, confidence scores, top-k results, and visual explanations through Grad-CAM.

---

##  Features

###  Interactive Drawing Interface
- Browser-based drawing canvas for seamless interaction
- Real-time digit submission and processing
- Responsive and intuitive user interface
- Clear button to start new predictions

###  Deep Learning Prediction
- CNN model trained on comprehensive handwritten digit datasets
- Instant digit classification with sub-second latency
- Probability-based confidence scoring
- Top-3 prediction ranking for model transparency

###  Explainable AI (XAI)
- Grad-CAM heatmap visualization
- Highlights image regions influencing predictions
- Improves model interpretability and transparency
- Helps users understand model decision-making

###  Model Analytics Dashboard
- Confusion matrix visualization for multi-class analysis
- Per-class accuracy metrics
- Overall model accuracy and loss statistics
- Training history tracking and visualization

###  Full-Stack Architecture
- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Flask REST API
- **ML Framework:** TensorFlow / Keras
- **Real-time Inference:** Optimized prediction pipeline

---

##  System Architecture

```
User Drawing
      ↓
Frontend (HTML/CSS/JS)
      ↓
Flask API Endpoint
      ↓
Image Preprocessing
      ↓
CNN Model Inference
      ├─ Predicted Digit
      ├─ Confidence Score
      ├─ Top-3 Predictions
      └─ Grad-CAM Heatmap
      ↓
Result Visualization & Display
```

---

##  Project Structure

```
real-time-digit-recognition/
│
├── backend/
│   ├── app.py                 # Flask application & routes
│   ├── utils.py               # Helper functions & preprocessing
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── templates/
│   │   └── index.html         # Main UI template
│   └── static/
│       ├── css/
│       │   └── styles.css     # Application styling
│       ├── js/
│       │   └── app.js         # Canvas & API interactions
│       └── assets/            # Images & resources
│
├── model/
│   └── saved_model/
│       ├── digit_cnn.keras    # Trained CNN model
│       ├── confusion_matrix.npy
│       ├── class_accuracy.json
│       └── training_history.json
│
└── README.md                  # Project documentation
```

---

##  Model Pipeline

### 1. Image Acquisition
Users draw handwritten digits directly on the browser canvas using mouse or touch input.

### 2. Image Preprocessing
- Base64 image decoding from canvas
- Grayscale color conversion
- Resizing to model input dimensions (typically 28×28)
- Pixel normalization (0-1 range)

### 3. CNN Inference
The preprocessed image is fed into the trained CNN model for classification.

### 4. Prediction Output
The system returns:
- **Predicted digit** – The most likely classification
- **Confidence score** – Probability percentage
- **Top-3 predictions** – Alternative classifications with scores
- **Grad-CAM heatmap** – Visual explanation of prediction

### 5. Analytics & Tracking
Performance metrics are retrieved from saved evaluation artifacts and displayed in the dashboard.

---

##  API Endpoints

### Predict Digit
**POST** `/predict`

**Request:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "predicted_digit": 8,
  "confidence": 99.24,
  "top_3_predictions": [
    {
      "digit": 8,
      "confidence": 99.24
    },
    {
      "digit": 3,
      "confidence": 0.42
    },
    {
      "digit": 6,
      "confidence": 0.18
    }
  ],
  "heatmap": "base64_encoded_heatmap"
}
```

### Model Analytics
**GET** `/analytics`

**Response:**
```json
{
  "confusion_matrix": [],
  "overall_accuracy": 0.99,
  "overall_loss": 0.02,
  "class_accuracy": {},
  "training_history": {}
}
```

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/real-time-digit-recognition.git
cd real-time-digit-recognition
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
cd backend
python app.py
```

### 6. Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

##  Key Features Implemented

 End-to-end ML application development  
 CNN-based multi-class digit classification  
 Real-time Flask API model serving  
 Grad-CAM explainable AI integration  
 Interactive canvas drawing interface  
 Comprehensive model performance analytics  
 Responsive frontend design  
 Base64 image encoding/decoding  

---

##  Future Improvements

- [ ] Multi-digit recognition and segmentation
- [ ] Cloud deployment (AWS, Render, Railway, Vercel)
- [ ] Model comparison and benchmark dashboard
- [ ] Mobile-responsive drawing experience
- [ ] User prediction history and logging
- [ ] TensorFlow Lite conversion for edge deployment
- [ ] Dark mode support
- [ ] Real-time training metrics visualization
- [ ] Support for digit datasets beyond MNIST
- [ ] Model quantization for faster inference

---

##  Technologies Used

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript, Canvas API |
| **Backend** | Flask, Python |
| **ML/DL** | TensorFlow, Keras, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Grad-CAM |
| **Data Format** | Base64, JSON |

---

##  Model Performance

- **Architecture:** Convolutional Neural Network (CNN)
- **Training Dataset:** MNIST or similar handwritten digit dataset
- **Inference Time:** <100ms per prediction
- **Model Size:** Optimized for real-time serving

---

##  Usage Example

1. **Open the application** in your web browser
2. **Draw a digit** (0-9) on the canvas using your mouse or touchpad
3. **Click "Submit"** to send the image to the server
4. **View results:**
   - Predicted digit with confidence score
   - Top-3 alternative predictions
   - Grad-CAM heatmap showing decision regions
5. **Click "Clear"** to draw another digit
6. **Check analytics** to view overall model performance

---

##  Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---


---

##  Author

**Robiul Hasan Jisan**

---

##  Support & Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the author directly.

---

##  Learning Resources

- [Keras CNN Documentation](https://keras.io/guides/functional_api/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Grad-CAM Explanation](https://arxiv.org/abs/1610.02055)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

**Last Updated:** 2026  
**Status:** Active Development ✨

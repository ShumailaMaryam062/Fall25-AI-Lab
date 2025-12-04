# 💻 Laptop Price Predictor API v2.0

An advanced Flask-based REST API for predicting laptop prices using machine learning with Random Forest algorithm.

## ✨ Key Features

- **Modern Web Interface** - Beautiful, responsive UI with dropdown selections
- **Multiple Prediction Modes** - Single, batch, and comparison predictions
- **REST API** - Complete API with multiple endpoints
- **Real-time Statistics** - Track prediction stats and usage
- **CORS Enabled** - Cross-origin requests supported
- **Request Timing** - Performance monitoring built-in
- **Comprehensive Logging** - File and console logging
- **Enhanced Validation** - Input validation with detailed error messages
- **Price Categories** - Automatic categorization (Budget/Mid-Range/Premium/High-End)

## 🚀 Quick Start
1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the application:**

```bash
python app.py
```

3. **Open your browser:**

Visit `http://localhost:5000`

## 🔌 API Endpoints

### Health & Info

#### GET `/health`
Check API status, model availability, and statistics.

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "version": "2.0.0",
  "stats": {
    "total_predictions": 150,
    "average_price": 65000.50
  }
}
```

#### GET `/features`
Get comprehensive feature documentation with examples.

#### GET `/api/stats`
Get usage statistics.

### Predictions

#### POST `/predict`
Single laptop price prediction (supports both JSON and form data).

**Request:**
```json
{
  "brand": 1,
  "processor_brand": 0,
  "processor_name": 1,
  "proc_gn": 11,
  "ram": 16,
  "ram_type": 1,
  "ssd": 512,
  "hdd": 0,
  "graphic_card_gb": 4
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 75000.00,
  "formatted_price": "₹75,000.00",
  "category": "Mid-Range",
  "currency": "INR",
  "confidence": "high"
}
```

#### POST `/api/batch-predict`
Batch predictions for multiple laptops (max 100 per request).

**Request:**
```json
{
  "laptops": [
    {
      "brand": 1,
      "processor_brand": 0,
      "...": "..."
    },
    { "...": "..." }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "summary": {
    "total": 2,
    "successful": 2,
    "failed": 0,
    "average_price": 70000.00
  },
  "predictions": [...]
}
```

#### POST `/api/compare`
Compare multiple laptop configurations side-by-side.

## 📝 Feature Specifications

| Feature | Type | Options/Range | Description |
|---------|------|---------------|-------------|
| `brand` | Categorical | 0-7 | Laptop brand (0=Dell, 1=HP, 2=Lenovo, etc.) |
| `processor_brand` | Categorical | 0-1 | Processor brand (0=Intel, 1=AMD) |
| `processor_name` | Categorical | 0-3 | Processor model (0=i3/R3, 1=i5/R5, 2=i7/R7, 3=i9/R9) |
| `proc_gn` | Numeric | 7-13 | Processor generation |
| `ram` | Numeric | 4, 8, 16, 32, 64 | RAM in GB |
| `ram_type` | Categorical | 0-2 | RAM type (0=DDR3, 1=DDR4, 2=DDR5) |
| `ssd` | Numeric | 0, 128, 256, 512, 1024, 2048 | SSD storage in GB |
| `hdd` | Numeric | 0, 500, 1000, 2000 | HDD storage in GB |
| `graphic_card_gb` | Numeric | 0, 2, 4, 6, 8, 12 | Graphics memory in GB |

## 💡 Example Usage

### Web Interface
Simply visit `http://localhost:5000` and use the intuitive form with dropdown selections.

### cURL
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": 1,
    "processor_brand": 0,
    "processor_name": 1,
    "proc_gn": 11,
    "ram": 16,
    "ram_type": 1,
    "ssd": 512,
    "hdd": 0,
    "graphic_card_gb": 4
  }'
```

### Python
```python
import requests

data = {
    "brand": 1,
    "processor_brand": 0,
    "processor_name": 1,
    "proc_gn": 11,
    "ram": 16,
    "ram_type": 1,
    "ssd": 512,
    "hdd": 0,
    "graphic_card_gb": 4
}

response = requests.post('http://localhost:5000/api/predict', json=data)
result = response.json()
print(f"Predicted Price: {result['formatted_price']}")
print(f"Category: {result['category']}")
```

### JavaScript
```javascript
const predictPrice = async (specs) => {
  const response = await fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(specs)
  });
  const result = await response.json();
  console.log(`Price: ${result.formatted_price}`);
  console.log(`Category: ${result.category}`);
};
```

## 🛠️ Tech Stack

- **Flask**: Web framework
- **Flask-CORS**: CORS support
- **scikit-learn**: Machine learning
- **NumPy**: Numerical computing

## 📄 License

MIT License

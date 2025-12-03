from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import logging
from datetime import datetime
import os
import json
from functools import wraps
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Load model with error handling
try:
    model = pickle.load(open('model_rf.pkl', 'rb'))
    logger.info('✓ Model loaded successfully')
except Exception as e:
    logger.error(f'✗ Error loading model: {e}')
    model = None

# Feature definitions with metadata
FEATURES = {
    'brand': {'type': 'categorical', 'description': 'Laptop brand', 'example': 1},
    'processor_brand': {'type': 'categorical', 'description': 'Processor brand (0=Intel, 1=AMD)', 'example': 0},
    'processor_name': {'type': 'categorical', 'description': 'Processor model', 'example': 5},
    'proc_gn': {'type': 'numeric', 'description': 'Processor generation', 'min': 7, 'max': 13, 'example': 11},
    'ram': {'type': 'numeric', 'description': 'RAM in GB', 'options': [4, 8, 16, 32, 64], 'example': 16},
    'ram_type': {'type': 'categorical', 'description': 'RAM type (0=DDR3, 1=DDR4, 2=DDR5)', 'example': 1},
    'ssd': {'type': 'numeric', 'description': 'SSD storage in GB', 'options': [0, 128, 256, 512, 1024, 2048], 'example': 512},
    'hdd': {'type': 'numeric', 'description': 'HDD storage in GB', 'options': [0, 500, 1000, 2000], 'example': 0},
    'graphic_card_gb': {'type': 'numeric', 'description': 'Graphics memory in GB', 'options': [0, 2, 4, 6, 8, 12], 'example': 4}
}

FEATURE_NAMES = list(FEATURES.keys())

# Statistics storage
prediction_stats = {
    'total_predictions': 0,
    'total_batch_predictions': 0,
    'average_price': 0,
    'last_prediction_time': None
}

# Middleware for request timing
def timing_decorator(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f'{f.__name__} took {(end_time - start_time)*1000:.2f}ms')
        return result
    return decorated_function

@app.route('/')
def home():
    """Home page with enhanced UI"""
    return render_template('index.html', features=FEATURES, feature_names=FEATURE_NAMES)

@app.route('/health', methods=['GET'])
@timing_decorator
def health_check():
    """Comprehensive health check endpoint"""
    model_status = 'loaded' if model is not None else 'not loaded'
    uptime = time.time()
    
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model': model_status,
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'stats': prediction_stats,
        'endpoints': {
            'web': '/',
            'health': '/health',
            'features': '/features',
            'predict': '/predict',
            'api_predict': '/api/predict',
            'batch_predict': '/api/batch-predict',
            'stats': '/api/stats'
        }
    }), 200 if model is not None else 503

@app.route('/predict', methods=['POST'])
@timing_decorator
def predict():
    """Enhanced prediction endpoint with validation and statistics"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'success': False
            }), 503
        
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Validate and extract features
        features = []
        validation_errors = []
        
        for feature_name in FEATURE_NAMES:
            if feature_name not in data:
                validation_errors.append(f'Missing: {feature_name}')
                continue
            
            try:
                value = float(data[feature_name])
                feature_info = FEATURES[feature_name]
                
                # Validate numeric ranges
                if 'min' in feature_info and value < feature_info['min']:
                    validation_errors.append(f'{feature_name} below minimum ({feature_info["min"]})')
                if 'max' in feature_info and value > feature_info['max']:
                    validation_errors.append(f'{feature_name} above maximum ({feature_info["max"]})')
                
                features.append(value)
            except ValueError:
                validation_errors.append(f'Invalid value for {feature_name}: {data[feature_name]}')
        
        if validation_errors:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_errors,
                'success': False
            }), 400
        
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        # Update statistics
        prediction_stats['total_predictions'] += 1
        prediction_stats['last_prediction_time'] = datetime.now().isoformat()
        if prediction_stats['average_price'] == 0:
            prediction_stats['average_price'] = prediction
        else:
            prediction_stats['average_price'] = (prediction_stats['average_price'] + prediction) / 2
        
        logger.info(f'✓ Prediction: ₹{prediction:,.2f}')
        
        # Determine price category
        if prediction < 30000:
            category = 'Budget'
        elif prediction < 60000:
            category = 'Mid-Range'
        elif prediction < 100000:
            category = 'Premium'
        else:
            category = 'High-End'
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'formatted_price': f'₹{prediction:,.2f}',
            'category': category,
            'currency': 'INR',
            'confidence': 'high',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f'✗ Prediction error: {str(e)}')
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'success': False
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Dedicated API endpoint for JSON requests with detailed response"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'required_features': FEATURE_NAMES,
                'success': False
            }), 400
        
        # Validate and extract features
        features = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                return jsonify({
                    'error': f'Missing feature: {feature}',
                    'required_features': FEATURE_NAMES,
                    'success': False
                }), 400
            try:
                features.append(float(data[feature]))
            except ValueError:
                return jsonify({
                    'error': f'Invalid value for feature {feature}',
                    'success': False
                }), 400

        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        logger.info(f'API prediction made: ₹{prediction:,.2f}')
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'formatted_price': f'₹{prediction:,.2f}',
            'currency': 'INR',
            'input_features': data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f'API prediction error: {str(e)}')
        return jsonify({
            'error': str(e),
            'required_features': FEATURE_NAMES,
            'success': False
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Comprehensive feature documentation"""
    return jsonify({
        'features': FEATURES,
        'feature_order': FEATURE_NAMES,
        'total_features': len(FEATURE_NAMES),
        'example_payloads': {
            'budget_laptop': {
                'brand': 1,
                'processor_brand': 0,
                'processor_name': 3,
                'proc_gn': 10,
                'ram': 8,
                'ram_type': 1,
                'ssd': 256,
                'hdd': 0,
                'graphic_card_gb': 0
            },
            'gaming_laptop': {
                'brand': 2,
                'processor_brand': 0,
                'processor_name': 7,
                'proc_gn': 12,
                'ram': 16,
                'ram_type': 2,
                'ssd': 512,
                'hdd': 1000,
                'graphic_card_gb': 6
            },
            'professional_laptop': {
                'brand': 3,
                'processor_brand': 0,
                'processor_name': 6,
                'proc_gn': 11,
                'ram': 32,
                'ram_type': 1,
                'ssd': 1024,
                'hdd': 0,
                'graphic_card_gb': 4
            }
        }
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics"""
    return jsonify({
        'statistics': prediction_stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/batch-predict', methods=['POST'])
@timing_decorator
def batch_predict():
    """Enhanced batch prediction with detailed analytics"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'success': False
            }), 503
        
        data = request.get_json()
        
        if not data or 'laptops' not in data:
            return jsonify({
                'error': 'Invalid request format',
                'expected_format': {'laptops': [{'brand': 1, 'processor_brand': 0, '...': '...'}]},
                'success': False
            }), 400
        
        laptops = data['laptops']
        
        if len(laptops) > 100:
            return jsonify({
                'error': 'Batch size too large',
                'max_batch_size': 100,
                'received': len(laptops),
                'success': False
            }), 400
        
        predictions = []
        successful = 0
        failed = 0
        total_price = 0
        
        for idx, laptop in enumerate(laptops):
            try:
                features = []
                for feature in FEATURE_NAMES:
                    if feature not in laptop:
                        predictions.append({
                            'index': idx,
                            'error': f'Missing feature: {feature}',
                            'status': 'failed'
                        })
                        failed += 1
                        continue
                    features.append(float(laptop[feature]))
                
                if len(features) != len(FEATURE_NAMES):
                    continue
                
                features_array = np.array(features).reshape(1, -1)
                prediction = model.predict(features_array)[0]
                
                predictions.append({
                    'index': idx,
                    'prediction': round(prediction, 2),
                    'formatted_price': f'₹{prediction:,.2f}',
                    'category': 'Budget' if prediction < 30000 else 'Mid-Range' if prediction < 60000 else 'Premium' if prediction < 100000 else 'High-End',
                    'status': 'success'
                })
                successful += 1
                total_price += prediction
                
            except Exception as e:
                predictions.append({
                    'index': idx,
                    'error': str(e),
                    'status': 'failed'
                })
                failed += 1
        
        # Update statistics
        prediction_stats['total_batch_predictions'] += successful
        prediction_stats['total_predictions'] += successful
        
        logger.info(f'✓ Batch prediction: {successful} successful, {failed} failed')
        
        return jsonify({
            'success': True,
            'summary': {
                'total': len(laptops),
                'successful': successful,
                'failed': failed,
                'average_price': round(total_price / successful, 2) if successful > 0 else 0
            },
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f'✗ Batch prediction error: {str(e)}')
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e),
            'success': False
        }), 500

@app.route('/api/compare', methods=['POST'])
@timing_decorator
def compare_laptops():
    """Compare multiple laptop configurations"""
    try:
        if model is None:
            return jsonify({'error': 'Model not available', 'success': False}), 503
        
        data = request.get_json()
        if not data or 'laptops' not in data or len(data['laptops']) < 2:
            return jsonify({
                'error': 'Please provide at least 2 laptops to compare',
                'success': False
            }), 400
        
        comparisons = []
        for laptop in data['laptops']:
            features = [float(laptop[f]) for f in FEATURE_NAMES]
            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            comparisons.append({
                'specs': laptop,
                'predicted_price': round(prediction, 2),
                'formatted_price': f'₹{prediction:,.2f}'
            })
        
        # Find best value
        best_value_idx = min(range(len(comparisons)), key=lambda i: comparisons[i]['predicted_price'])
        comparisons[best_value_idx]['recommendation'] = 'Best Value'
        
        return jsonify({
            'success': True,
            'comparisons': comparisons,
            'best_value_index': best_value_idx
        })
    
    except Exception as e:
        logger.error(f'✗ Comparison error: {str(e)}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.errorhandler(404)
def not_found(error):
    """Enhanced 404 handler"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 404,
        'available_endpoints': {
            'GET': ['/health', '/features', '/api/stats'],
            'POST': ['/predict', '/api/predict', '/api/batch-predict', '/api/compare']
        },
        'documentation': '/features'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Enhanced 500 handler"""
    logger.error(f'✗ Internal server error: {error}')
    return jsonify({
        'error': 'Internal server error',
        'status': 500,
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }), 500

@app.errorhandler(413)
def request_too_large(error):
    """Handle request size errors"""
    return jsonify({
        'error': 'Request too large',
        'max_size': '16MB',
        'status': 413
    }), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print('\n' + '='*60)
    print('🚀 LAPTOP PRICE PREDICTOR API v2.0')
    print('='*60)
    logger.info(f'🌐 Server starting on http://localhost:{port}')
    logger.info(f'🔧 Debug mode: {debug}')
    logger.info(f'📊 Model status: {"Loaded ✓" if model else "Not loaded ✗"}')
    print('\n📌 Available Endpoints:')
    print('   • GET  /              - Web Interface')
    print('   • GET  /health        - Health Check')
    print('   • GET  /features      - API Documentation')
    print('   • GET  /api/stats     - Usage Statistics')
    print('   • POST /predict       - Single Prediction')
    print('   • POST /api/predict   - API Prediction')
    print('   • POST /api/batch-predict - Batch Predictions')
    print('   • POST /api/compare   - Compare Laptops')
    print('='*60 + '\n')
    
    app.run(debug=debug, host='0.0.0.0', port=port)

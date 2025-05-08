from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models and preprocessing objects
try:
    rf_model = joblib.load('rf_model.pkl')
    ann_model = joblib.load('ann_model.pkl')
    scaler = joblib.load('scaler.pkl')
    protocol_encoder = joblib.load('protocol_type_encoder.pkl')
    service_encoder = joblib.load('service_encoder.pkl')
    flag_encoder = joblib.load('flag_encoder.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Feature list (must match training)
features = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'count', 'srv_count',
    'same_srv_rate', 'dst_host_srv_count'
]

def get_prediction_result(proba):
    """Handle single-class or multi-class probability arrays"""
    if len(proba) == 1:  # Only one class exists
        return "Normal" if proba[0] > 0.5 else "Attack"
    else:  # Two classes exist
        return "Attack" if proba[1] > 0.5 else "Normal"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            model_choice = form_data.pop('model_choice', 'both')
            
            # Create feature vector
            input_features = []
            for feature in features:
                if feature in ['protocol_type', 'service', 'flag']:
                    encoder = {
                        'protocol_type': protocol_encoder,
                        'service': service_encoder,
                        'flag': flag_encoder
                    }[feature]
                    input_features.append(encoder.transform([form_data[feature]])[0])
                else:
                    input_features.append(float(form_data[feature]))
            
            input_array = np.array(input_features).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            
            results = {
                'rf_result': None,
                'rf_confidence': None,
                'ann_result': None,
                'ann_confidence': None
            }
            
            if model_choice in ['rf', 'both']:
                rf_proba = rf_model.predict_proba(scaled_input)[0]
                results['rf_result'] = get_prediction_result(rf_proba)
                results['rf_confidence'] = round(max(rf_proba) * 100, 1)
            
            if model_choice in ['ann', 'both']:
                ann_proba = ann_model.predict_proba(scaled_input)[0]
                results['ann_result'] = get_prediction_result(ann_proba)
                results['ann_confidence'] = round(max(ann_proba) * 100, 1)
            
            return render_template('results.html',
                                model_choice=model_choice,
                                input_data={k:v for k,v in form_data.items() if k != 'model_choice'},
                                **results)
        
        except Exception as e:
            return render_template('error.html', error_message=str(e))
    
    # GET request - show form
    try:
        return render_template('predict.html',
                            protocol_types=list(protocol_encoder.classes_),
                            services=list(service_encoder.classes_),
                            flags=list(flag_encoder.classes_))
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
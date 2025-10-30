# pip install flask flask-cors
from flask import Flask, request, jsonify
from flask_cors import CORS
from services.job_orchestrator import JobOrchestrator
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Instantiate the core orchestrator once
orchestrator = JobOrchestrator()

@app.route('/api/process', methods=['POST'])
def process_video():
    data = request.json
    youtube_url = data.get('url')
    
    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    job_id = orchestrator.start_job(youtube_url)
    
    return jsonify({'job_id': job_id, 'status': 'processing'}), 202

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = orchestrator.get_status(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    response = {
        'id': job['id'],
        'status': job['status'],
        'progress': job.get('progress', 0),
        'message': job.get('message', '')
    }
    
    if job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    job = orchestrator.get_result(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    # Check if the job finished via Direct Extraction or ML Pipeline
    if job['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'topics': job['topics']
        })

    # If status is not 'completed', return 202 accepted (still processing)
    return jsonify({
        'status': job['status'],
        'message': 'Processing not completed yet'
    }), 202


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'active_jobs': orchestrator.get_active_jobs(),
    })

if __name__ == '__main__':
    print("Starting YouTube Topic Segmentation Server (Intelligent Hybrid Architecture)...")
    print("Server will be available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
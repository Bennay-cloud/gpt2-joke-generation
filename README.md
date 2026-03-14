# GPT-2 Joke Generator – End-to-End MLOps Pipeline

This project implements an end-to-end MLOps pipeline that fine-tunes a GPT-2 language model on a jokes dataset and deploys it as a scalable inference service.

# The system includes:
	•	GPT-2 fine-tuning with HuggingFace Transformers
	•	automated model evaluation with a quality threshold
	•	model versioning using Azure Blob Storage
	•	containerization with Docker
	•	deployment to Azure Kubernetes Service (AKS)
	•	CI/CD automation using GitHub Actions
	•	a Flask REST API for inference
	•	a Streamlit web UI for user interaction

Users can generate jokes by submitting prompts through the web interface or API.

⸻

# Tech Stack
	•	Python
	•	HuggingFace Transformers
	•	PyTorch
	•	Docker
	•	Kubernetes (AKS)
	•	Azure Container Registry
	•	Azure Blob Storage
	•	GitHub Actions
	•	Flask
	•	Streamlit

⸻

# Project Structure

GPT2_AllTrans/
│
├── app/                # Inference API
│   ├── score.py
│   ├── model/
│   ├── tokenizer/
│   └── requirements.txt
│
├── train/              # Training pipeline
│   ├── train.py
│   ├── evaluate.py
│   └── promote_model.py
│
├── deploy/             # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
│
├── webapp/             # Streamlit interface
│   └── app.py
│
└── README.md


⸻

# Installation

1. Clone the repository

git clone https://github.com/<your-username>/GPT2_AllTrans.git
cd GPT2_AllTrans

2. Create a virtual environment

python -m venv venv
source venv/bin/activate

On Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r app/requirements.txt


⸻

# Run the Inference API

Start the Flask API:

python app/score.py

The API will run at:

http://localhost:5001

Health check:

curl http://localhost:5001/health

Generate a joke:

curl -X POST http://localhost:5001/joke \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Why did the programmer quit his job?"}'


⸻

# Run the Streamlit Web UI

Start the interface:

streamlit run webapp/app.py

Open in your browser:

http://localhost:8501

Users can enter prompts and receive generated jokes from the GPT-2 model.

⸻

# Example Prompt

Why did the programmer quit his job?


⸻

Future Improvements
	•	automated retraining pipeline
	•	model monitoring
	•	A/B testing
	•	larger training datasets
	•	GPU training support



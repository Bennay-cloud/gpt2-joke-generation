flowchart LR

A[Dataset] --> B[Training Pipeline]
B --> C[MLflow Tracking]
B --> D[Model Artifacts]

D --> E[Azure Blob Storage]

E --> F[GitHub Actions CI/CD]

F --> G[Docker Build]

G --> H[Azure Container Registry]

H --> I[Azure Kubernetes Service]

I --> J[Flask Inference API]

J --> K[Streamlit Web UI]

K --> L[User]

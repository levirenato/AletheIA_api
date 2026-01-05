# Aletheia Engine: Brazilian Identity Document Biometric Verification

This project provides a high-performance facial verification and document validation engine optimized for Python (PyPy). It combines deep learning models via ONNX Runtime to ensure a Brazilian identity document is valid, unique, and matches the user's selfie.

---

## Technical Workflow

The engine utilizes a hierarchical validation pipeline to ensure security and computational efficiency:

1. **Document Classification**: A trained MobileNetV3 Small model performs initial screening to verify if the uploaded image is a valid Brazilian ID (CNH or RG). This prevents non-document images from consuming further processing resources.
2. **Robust Face Detection**: The engine uses the UltraFace model to locate human faces. It automatically attempts four rotations (0°, 90°, 180°, 270°) to find the correct orientation.
3. **Uniqueness Constraint**: A strict rule is applied where the document must contain exactly one face. Detection of multiple faces or zero faces results in immediate rejection.
4. **Anti-Fraud Layer (LEANN)**: The engine extracts a 512-D biometric embedding using ArcFace. It compares the document's face against a local vector database to prevent the same document from being used across multiple accounts.
5. **Biometric Matching**: If the document is valid and unique, the user's selfie is processed. A cosine similarity score is calculated between the document face and the selfie face. A successful match requires a score above 0.55.

---

## Project Structure

- `AletheiaEngine.py`: Core logic containing the validation pipeline and ONNX inference calls.
- `app.py`: Entry point for local execution and CLI testing.
- `models/`: Directory containing pre-trained ONNX models:
- `ultraface.onnx`: Face detection.
- `arcface.onnx`: Feature extraction (embeddings).
- `document_classifier.onnx`: MobileNetV3 document screening.

- `data/`: Persistent storage for the biometric vector database.
- `debug/`: Output directory for visual audit logs and comparison results.

---

## Setup and Execution with UV

This project uses **uv** for dependency management, ensuring fast and reproducible environments.

### Installation

To install dependencies and create a virtual environment, run:

```bash
uv venv
uv pip install opencv-python-headless numpy onnxruntime

```

### Running the Project

Ensure your model files are placed in the `models` folder and your test images in the `Examples` folder. Execute the script using:

```bash
uv run app.py

```

---

## Current Status and Roadmap

### Completed Features

- MobileNetV3 document classification.
- Automated image rotation handling.
- Biometric 1:1 face comparison.
- Local vector storage for duplicate detection.
- Visual debug generation through OpenCV.

### Pending Implementation

- **REST API Layer**: Integration with Socketify.py or FastAPI to expose endpoints for web and mobile clients.
- **Multi-part/form-data Handling**: Logic to process image uploads directly from HTTP requests.
- **Asynchronous Database Integration**: Moving from local storage to a professional vector database for high-concurrency environments.

---

## Academic Information

This project was developed as a university assignment focused on Applied Computer Vision and Document Intelligence. It demonstrates the practical application of lightweight neural networks in biometric security systems.

---

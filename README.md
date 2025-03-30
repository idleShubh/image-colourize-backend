# Image Colorization Service

A service that colorizes black and white images using deep learning.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/idleShubh/image_colorizer.git
cd image_colorizer
```

2. Download the required model files:
   - Visit [Google Drive Link](https://drive.google.com/drive/folders/1FaDajjtAsntF_Sw5gqF0WyakviA5l8-a)
   - Download the following files:
     - `colorization_release_v2.caffemodel` (123 MB)
     - `colorization_deploy_v2.prototxt` (10 KB)
     - `pts_in_hull.npy` (5 KB)
   - Create a `model` directory in the project root
   - Place all downloaded files in the `model` directory

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your database configuration
```

5. Start the backend server:
```bash
cd backend
uvicorn api:app --reload
```

6. Start the frontend development server:
```bash
cd frontend
npm install
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

## API Documentation

Once the server is running, visit http://localhost:8000/docs for the interactive API documentation.

## Features

- Upload black and white images
- Automatic colorization using deep learning
- Download colorized images
- Responsive web interface
- Real-time processing status
- Error handling and validation

## Technologies Used

- Backend: FastAPI, OpenCV, NumPy
- Frontend: React, Tailwind CSS
- Database: PostgreSQL
- Image Processing: OpenCV, NumPy
- Deep Learning: Caffe model for image colorization 
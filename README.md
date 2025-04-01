# MemoraPix - FaceLens Application

A facial recognition and photo organization system built with Django and React.

## Overview

MemoraPix is an application that helps users organize and search through their photo collections using facial recognition technology. The system automatically detects faces in uploaded photos, clusters similar faces together, and allows users to search their photo collection by person.

## Features

- Photo upload and management
- Automatic face detection and extraction
- Face clustering using machine learning
- Cluster naming and organization
- Photo search by person, date, or location
- Batch processing capabilities
- Interactive photo gallery

## Tech Stack

### Backend
- Django
- PostgreSQL
- DeepFace (for facial recognition)
- OpenCV (for image processing)
- scikit-learn (for clustering)

### Frontend
- React
- Axios (for API communication)
- CSS Modules

## Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- PostgreSQL
- OpenCV dependencies

### Backend Setup

1. Navigate to FaceLensBackend directory:
```bash
cd FaceLensBackend
```

2. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
DB_NAME=facelens
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=127.0.0.1
DB_PORT=5432
```

5. Create the media directory:
```bash
mkdir media
```

6. Run migrations:
```bash
python manage.py migrate
```

7. Start the Django development server:
```bash
python manage.py runserver
```
The backend will be available at http://127.0.0.1:8000

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a .env file:
```bash
REACT_APP_API_URL=http://127.0.0.1:8000
```

4. Start the development server:
```bash
npm start
```
The frontend will be available at http://localhost:3000

### Running Both Servers

You'll need two terminal windows:

Terminal 1 (Backend):
```bash
cd FaceLensBackend
source env/Scripts/activate  # On Windows
python manage.py runserver
```

Terminal 2 (Frontend):
```bash
cd frontend
npm start
```

Access the application at http://localhost:3000

### Common Issues

1. If you get CORS errors:
   - Make sure both servers are running
   - Check that the CORS settings in Django settings.py are correct
   - Verify the API_URL in frontend .env is correct

2. If face detection fails:
   - Ensure OpenCV is properly installed
   - Check that the media directory exists and is writable
   - Verify the image format is supported (JPG, PNG)

3. If clustering doesn't work:
   - Make sure scikit-learn is installed
   - Verify that there are enough faces detected for clustering
   - Check the PostgreSQL connection

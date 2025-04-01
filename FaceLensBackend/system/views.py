from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings  # Add this import
from .models import PHOTOS, FACES, FACE_CLUSTERS
import os
import datetime
from deepface import DeepFace
import cv2  # Import OpenCV for face detection
import numpy as np
from sklearn.cluster import DBSCAN
from django.db.models import Q
import uuid  # Import for generating unique names

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image = request.FILES['image']
            extension = os.path.splitext(image.name)[1]  # Get the file extension
            unique_name = f"{uuid.uuid4()}{extension}"  # Generate a unique name

            # Save the file directly in media folder without any prefix
            file_path = default_storage.save(unique_name, ContentFile(image.read()))
            upload_date = datetime.datetime.now()

            # Save metadata to the PHOTOS table
            photo = PHOTOS.objects.create(
                FILE_NAME=unique_name,
                FILE_PATH=file_path,  # Save the file path as is
                UPLOAD_DATE=upload_date
            )

            return JsonResponse({'message': 'Image uploaded successfully', 'record_id': photo.RECORD_ID}, status=200)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def process_image(request):
    if request.method == 'POST':
        try:
            record_id = request.POST.get('record_id')
            if not record_id:
                return JsonResponse({'error': 'Record ID is required'}, status=400)

            # Fetch the photo record
            photo = PHOTOS.objects.get(RECORD_ID=record_id)
            
            # Construct absolute path using MEDIA_ROOT
            image_path = os.path.join(settings.MEDIA_ROOT, photo.FILE_PATH)
            print(f"Processing image at path: {image_path}")

            # Verify file exists
            if not os.path.exists(image_path):
                return JsonResponse({'error': f'Image not found at {image_path}'}, status=404)

            # Load the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return JsonResponse({'error': 'Could not read image file'}, status=400)

            # Rest of face detection and processing code...
            
            # Add debug logging
            print(f"Processing image: {image_path}")
            print(f"Image shape: {image.shape if image is not None else 'None'}")

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Load OpenCV's pre-trained Haar Cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(60, 60),
                maxSize=(800, 800)
            )

            print(f"Detected {len(faces)} faces")

            # Process each detected face
            faces_processed = 0
            for (x, y, w, h) in faces:
                try:
                    # Add padding around the face
                    pad = int(max(w, h) * 0.2)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(image.shape[1], x + w + pad)
                    y2 = min(image.shape[0], y + h + pad)
                    
                    face = image[y1:y2, x1:x2]
                    
                    if face.shape[0] < 64 or face.shape[1] < 64:
                        continue

                    face_embedding = DeepFace.represent(
                        img_path=face,
                        model_name='VGG-Face',
                        enforce_detection=False,
                        detector_backend='opencv'
                    )[0]['embedding']

                    FACES.objects.create(
                        PHOTO=photo,
                        FACE_LOCATION={
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h)
                        },
                        FACE_EMBEDDING=face_embedding,
                        DETECTION_CONFIDENCE=1.0
                    )
                    faces_processed += 1

                except Exception as e:
                    print(f"Error processing face in {photo.FILE_NAME}: {str(e)}")
                    continue

            return JsonResponse({
                'message': f'Successfully processed {faces_processed} faces in image',
                'faces_detected': len(faces),
                'faces_processed': faces_processed
            })

        except PHOTOS.DoesNotExist:
            return JsonResponse({'error': f'Photo record not found for ID: {record_id}'}, status=404)
        except Exception as e:
            print(f"Error processing image: {str(e)}")  # Add server-side logging
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def cluster_faces(request):
    if request.method == 'POST':
        try:
            # Fetch all face embeddings from the database
            faces = FACES.objects.filter(CLUSTER__isnull=True).values('RECORD_ID', 'FACE_EMBEDDING')
            if not faces:
                return JsonResponse({'message': 'No unclustered faces found.'})

            embeddings = [face['FACE_EMBEDDING'] for face in faces]
            record_ids = [face['RECORD_ID'] for face in faces]

            print(f"Embeddings: {embeddings}")  # Debugging log
            print(f"Record IDs: {record_ids}")  # Debugging log

            # Perform clustering using DBSCAN
            dbscan = DBSCAN(
                eps=0.4,              # Reduced epsilon for stricter clustering
                min_samples=3,        # Increased minimum samples
                metric='cosine',      # Changed to cosine similarity
                n_jobs=-1             # Use all CPU cores
            )
            
            # Normalize embeddings before clustering
            embeddings_array = np.array(embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1)
            normalized_embeddings = embeddings_array / norms[:, np.newaxis]
            
            cluster_labels = dbscan.fit_predict(normalized_embeddings)

            print(f"Cluster Labels: {cluster_labels}")  # Debugging log

            # Save clusters to the database
            for record_id, cluster_id in zip(record_ids, cluster_labels):
                if cluster_id != -1:  # Ignore noise points
                    cluster, created = FACE_CLUSTERS.objects.get_or_create(NAME=f'Cluster {cluster_id}')
                    FACES.objects.filter(RECORD_ID=record_id).update(CLUSTER=cluster)

            # Count faces in each cluster
            clusters = FACE_CLUSTERS.objects.all()
            for cluster in clusters:
                face_count = FACES.objects.filter(CLUSTER=cluster).count()
                cluster.FACE_COUNT = face_count
                cluster.save()

            return JsonResponse({
                'message': 'Clustering completed successfully.',
                'clusters_created': len(set(label for label in cluster_labels if label != -1))
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def assign_cluster_name(request):
    if request.method == 'POST':
        try:
            cluster_id = request.POST.get('cluster_id')
            name = request.POST.get('name')

            if not cluster_id or not name:
                return JsonResponse({'error': 'Cluster ID and name are required.'}, status=400)

            # Update the cluster name
            cluster = FACE_CLUSTERS.objects.get(RECORD_ID=cluster_id)
            cluster.NAME = name
            cluster.save()

            return JsonResponse({'message': f'Cluster {cluster_id} renamed to {name}.'})

        except FACE_CLUSTERS.DoesNotExist:
            return JsonResponse({'error': f'Cluster with ID {cluster_id} does not exist.'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def search_images(request):
    if request.method == 'GET':
        try:
            cluster_name = request.GET.get('cluster_name')
            face_id = request.GET.get('face_id')
            start_date = request.GET.get('start_date')
            end_date = request.GET.get('end_date')
            location = request.GET.get('location')

            # Base query for photos
            photos_query = PHOTOS.objects.all()

            # Filter by cluster name
            if cluster_name:
                clusters = FACE_CLUSTERS.objects.filter(NAME__icontains=cluster_name)
                faces = FACES.objects.filter(CLUSTER__in=clusters)
                photos_query = photos_query.filter(RECORD_ID__in=faces.values('PHOTO'))

            # Filter by face ID
            if face_id:
                faces = FACES.objects.filter(RECORD_ID=face_id)
                photos_query = photos_query.filter(RECORD_ID__in=faces.values('PHOTO'))

            # Filter by date range
            if start_date and end_date:
                photos_query = photos_query.filter(UPLOAD_DATE__range=[start_date, end_date])

            # Filter by location
            if location:
                photos_query = photos_query.filter(LOCATION__icontains=location)

            # Serialize and return the results
            photos = list(photos_query.values('RECORD_ID', 'FILE_NAME', 'FILE_PATH', 'UPLOAD_DATE', 'LOCATION'))
            return JsonResponse({'photos': photos})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def process_all_photos(request):
    if request.method == 'POST':
        try:
            # Get all photos
            unprocessed_photos = PHOTOS.objects.all()
            processed_count = 0

            for photo in unprocessed_photos:
                try:
                    # Load and process each image
                    image_path = os.path.join('media', photo.FILE_PATH)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        continue

                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    # Process each detected face
                    for (x, y, w, h) in faces:
                        face_embedding = DeepFace.represent(img_path=image_path, enforce_detection=False)[0]['embedding']
                        
                        face_location = {
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h)
                        }

                        FACES.objects.create(
                            PHOTO=photo,
                            FACE_LOCATION=face_location,
                            FACE_EMBEDDING=face_embedding,
                            DETECTION_CONFIDENCE=1.0
                        )
                    
                    processed_count += 1

                except Exception as e:
                    print(f"Error processing photo {photo.FILE_NAME}: {str(e)}")
                    continue

            # Run clustering after processing all photos
            faces = FACES.objects.filter(CLUSTER__isnull=True).values('RECORD_ID', 'FACE_EMBEDDING')
            if faces:
                embeddings = [face['FACE_EMBEDDING'] for face in faces]
                record_ids = [face['RECORD_ID'] for face in faces]

                dbscan = DBSCAN(eps=0.8, min_samples=2, metric='euclidean')
                cluster_labels = dbscan.fit_predict(embeddings)

                # Save clusters
                for record_id, cluster_id in zip(record_ids, cluster_labels):
                    if cluster_id != -1:
                        cluster, created = FACE_CLUSTERS.objects.get_or_create(NAME=f'Cluster {cluster_id}')
                        FACES.objects.filter(RECORD_ID=record_id).update(CLUSTER=cluster)

            return JsonResponse({
                'message': f'Processed {processed_count} photos and created clusters successfully.',
                'processed_count': processed_count
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def get_clusters(request):
    if request.method == 'GET':
        try:
            clusters = FACE_CLUSTERS.objects.all()
            clusters_data = list(clusters.values('RECORD_ID', 'NAME', 'FACE_COUNT'))
            return JsonResponse({'clusters': clusters_data})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)
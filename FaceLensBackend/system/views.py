from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from .models import PHOTOS, FACES, FACE_CLUSTERS
import os
import datetime
from deepface import DeepFace
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from django.db.models import Q
import uuid
from mtcnn import MTCNN
import tensorflow as tf
from PIL import Image  # Add this import

def compare_faces(embedding1, embedding2, threshold=0.6):
    """Compare two face embeddings and return True if they are likely the same person"""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    return similarity > threshold

def preprocess_face(face_img):
    """Preprocess face for consistent embeddings regardless of color/grayscale"""
    # Convert to grayscale for consistency
    if len(face_img.shape) == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    else:
        face_gray = face_img
        
    # Normalize lighting
    face_gray = cv2.equalizeHist(face_gray)
    
    # Convert back to RGB (required for VGG-Face)
    face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
    
    # Resize to standard size
    face_rgb = cv2.resize(face_rgb, (224, 224))
    
    return face_rgb

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

            photo = PHOTOS.objects.get(RECORD_ID=record_id)
            image_path = os.path.join(settings.MEDIA_ROOT, photo.FILE_PATH)
            
            if not os.path.exists(image_path):
                return JsonResponse({'error': f'Image not found at {image_path}'}, status=404)

            # Initialize MTCNN with default parameters (removed invalid parameters)
            detector = MTCNN()

            # Load and process image
            img_array = cv2.imread(image_path)
            if img_array is None:
                return JsonResponse({'error': 'Could not read image file'}, status=400)

            # Resize large images to improve performance
            max_dimension = 1200
            height, width = img_array.shape[:2]
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                img_array = cv2.resize(img_array, (new_width, new_height))

            # Convert BGR to RGB (MTCNN expects RGB)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # Detect faces with MTCNN
            detected_faces = detector.detect_faces(img_array)
            faces_processed = 0

            for face_info in detected_faces:
                try:
                    # Only process faces with high confidence
                    if face_info['confidence'] < 0.95:
                        continue

                    # Get face area with margin
                    x, y, w, h = face_info['box']
                    margin = int(min(w, h) * 0.3)
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(img_array.shape[1] - x, w + 2 * margin)
                    h = min(img_array.shape[0] - y, h + 2 * margin)

                    # Skip if face is too small
                    if w < 60 or h < 60:  # Reduced minimum size threshold
                        continue
                    
                    # Extract face region
                    face_img = img_array[y:y+h, x:x+w]

                    # Ensure face image is not empty
                    if face_img.size == 0:
                        continue

                    # Save face image
                    face_filename = f"face_{str(uuid.uuid4())[:8]}.jpg"
                    face_path = os.path.join(settings.MEDIA_ROOT, 'faces', face_filename)
                    os.makedirs(os.path.dirname(face_path), exist_ok=True)
                    
                    # Convert RGB back to BGR for saving
                    face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(face_path, face_img_bgr)

                    try:
                        # Get embedding using DeepFace
                        embedding = DeepFace.represent(
                            img_path=face_path,
                            model_name='VGG-Face',
                            enforce_detection=False,
                            detector_backend='skip',
                            align=True
                        )

                        if embedding:
                            face_location = {
                                'x': int(x),
                                'y': int(y),
                                'width': int(w),
                                'height': int(h),
                                'face_image': f"faces/{face_filename}"
                            }

                            # Normalize embedding
                            embedding_array = np.array(embedding[0]['embedding'])
                            embedding_norm = embedding_array / np.linalg.norm(embedding_array)

                            # Save to database
                            FACES.objects.create(
                                PHOTO=photo,
                                FACE_LOCATION=face_location,
                                FACE_EMBEDDING=embedding_norm.tolist(),
                                DETECTION_CONFIDENCE=face_info['confidence']
                            )

                            faces_processed += 1

                    except Exception as embed_error:
                        print(f"Error getting embedding: {str(embed_error)}")
                        continue

                except Exception as face_error:
                    print(f"Error processing face: {str(face_error)}")
                    continue

            # After processing faces, trigger clustering
            if faces_processed > 0:
                cluster_faces(None)

            return JsonResponse({
                'message': f'Successfully processed {faces_processed} faces in image',
                'faces_detected': len(detected_faces),
                'faces_processed': faces_processed
            })

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def calculate_iou(box1, box2):
    """Calculate intersection over union between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

@csrf_exempt
def cluster_faces(request=None):
    """
    Improved clustering with better parameters and face matching
    """
    try:
        # Get all faces, including previously clustered ones
        all_faces = FACES.objects.all()
        if not all_faces.exists():
            return JsonResponse({'message': 'No faces found.'}) if request else None

        # Prepare embeddings
        embeddings = []
        face_ids = []
        
        for face in all_faces:
            embedding = np.array(face.FACE_EMBEDDING)
            embeddings.append(embedding)
            face_ids.append(face.RECORD_ID)

        embeddings_array = np.array(embeddings)

        # Improved DBSCAN parameters
        clustering = DBSCAN(
            eps=0.45,  # Increased to be more lenient in cluster formation
            min_samples=2,  # Reduced to allow smaller clusters
            metric='cosine',
            n_jobs=-1
        ).fit(embeddings_array)

        labels = clustering.labels_

        # Clear existing clusters
        FACE_CLUSTERS.objects.all().delete()

        # Process clustering results
        unique_labels = set(labels)
        clusters_created = 0
        
        for label in unique_labels:
            if label != -1:  # Ignore noise points
                # Create new cluster
                cluster = FACE_CLUSTERS.objects.create(
                    NAME=f'Person_{clusters_created}',
                    FACE_COUNT=0
                )
                clusters_created += 1

                # Assign faces to cluster
                mask = labels == label
                cluster_face_ids = [face_ids[i] for i, is_member in enumerate(mask) if is_member]
                
                # Update faces with new cluster
                FACES.objects.filter(RECORD_ID__in=cluster_face_ids).update(CLUSTER=cluster)
                
                # Update face count
                cluster.FACE_COUNT = len(cluster_face_ids)
                cluster.save()

        # Count results
        total_clustered = sum(1 for label in labels if label != -1)
        total_noise = sum(1 for label in labels if label == -1)

        result = {
            'message': 'Clustering completed successfully',
            'clustered_faces': total_clustered,
            'noise_faces': total_noise,
            'total_clusters': clusters_created
        }

        return JsonResponse(result) if request else result

    except Exception as e:
        error_response = {'status': 'error', 'message': str(e)}
        return JsonResponse(error_response) if request else error_response

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

            # Modified query to get all photos containing faces from this cluster
            if (cluster_name):
                # Find the cluster
                clusters = FACE_CLUSTERS.objects.filter(NAME__icontains=cluster_name)
                if clusters.exists():
                    # Get all faces in this cluster
                    cluster_faces = FACES.objects.filter(CLUSTER__in=clusters)
                    # Get all photos that contain these faces
                    photos_query = PHOTOS.objects.filter(
                        RECORD_ID__in=cluster_faces.values('PHOTO')
                    ).distinct()  # Use distinct to avoid duplicates
                    
                    # For each photo, get all faces detected in it
                    photos = []
                    for photo in photos_query:
                        faces_in_photo = FACES.objects.filter(PHOTO=photo).select_related('CLUSTER')
                        photo_data = {
                            'RECORD_ID': photo.RECORD_ID,
                            'FILE_NAME': photo.FILE_NAME,
                            'FILE_PATH': photo.FILE_PATH,
                            'UPLOAD_DATE': photo.UPLOAD_DATE,
                            'LOCATION': photo.LOCATION,
                            'faces': [{
                                'face_location': face.FACE_LOCATION,
                                'cluster_name': face.CLUSTER.NAME if face.CLUSTER else None,
                                'detection_confidence': face.DETECTION_CONFIDENCE
                            } for face in faces_in_photo]
                        }
                        photos.append(photo_data)
                    
                    return JsonResponse({'photos': photos})

            # Base query for photos
            photos_query = PHOTOS.objects.all()

            # Filter by cluster name
            if cluster_name:
                clusters = FACE_CLUSTERS.objects.filter(NAME__icontains=cluster_name)
                faces = FACES.objects.filter(CLUSTER__in=clusters)
                photos_query = photos_query.filter(RECORD_ID__in=faces.values('PHOTO'))

            # Filter by face ID
            face_id = request.GET.get('face_id')
            if face_id:
                faces = FACES.objects.filter(RECORD_ID=face_id)
                photos_query = photos_query.filter(RECORD_ID__in=faces.values('PHOTO'))

            # Filter by date range
            start_date = request.GET.get('start_date')
            end_date = request.GET.get('end_date')
            if start_date and end_date:
                photos_query = photos_query.filter(UPLOAD_DATE__range=[start_date, end_date])

            # Filter by location
            location = request.GET.get('location')
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

                dbscan = DBSCAN(
                    eps=0.35,
                    min_samples=2,
                    metric='cosine'
                )
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
            clusters_data = []
            
            for cluster in clusters:
                # Get a representative face for each cluster
                representative_face = FACES.objects.filter(CLUSTER=cluster).first()
                face_image_path = representative_face.FACE_LOCATION.get('face_image') if representative_face else None
                
                clusters_data.append({
                    'RECORD_ID': cluster.RECORD_ID,
                    'NAME': cluster.NAME,
                    'FACE_COUNT': cluster.FACE_COUNT,
                    'representative_face': face_image_path
                })
            
            return JsonResponse({'clusters': clusters_data})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request'}, status=400)
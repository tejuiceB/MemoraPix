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

            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                return JsonResponse({'error': 'Could not read image file'}, status=400)

            # Resize large images before processing
            max_dimension = 1200
            img_array = np.array(Image.open(image_path))
            height, width = img_array.shape[:2]
            
            if width > max_dimension or height > max_dimension:
                # Calculate new dimensions while maintaining aspect ratio
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                img_array = np.array(Image.fromarray(img_array).resize((new_width, new_height)))

            # Convert to RGB if needed
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

            # Initialize MTCNN detector
            detector = MTCNN()
            detected_faces = detector.detect_faces(img_array)

            faces_processed = 0
            for face_info in detected_faces:
                try:
                    if face_info['confidence'] < 0.85:
                        continue

                    # Convert all numeric values to standard Python types
                    x, y, w, h = map(int, face_info['box'])  # Convert to regular Python int
                    
                    face = img_array[y:y+h, x:x+w]
                    face = cv2.resize(face, (224, 224))
                    
                    # Generate unique filename for the face
                    face_filename = f"face_{uuid.uuid4()}.jpg"
                    face_path = os.path.join(settings.MEDIA_ROOT, 'faces', face_filename)
                    
                    # Ensure faces directory exists
                    os.makedirs(os.path.join(settings.MEDIA_ROOT, 'faces'), exist_ok=True)
                    
                    # Save face image
                    cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                    # Get embedding
                    embedding = DeepFace.represent(
                        img_path=face_path,
                        model_name='VGG-Face',
                        enforce_detection=False,
                        detector_backend='skip'
                    )

                    if embedding:
                        # Convert keypoints to serializable format
                        keypoints = {
                            k: [int(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) 
                            else int(v) 
                            for k, v in face_info['keypoints'].items()
                        }

                        # Convert embedding to regular Python list with float values
                        embedding_list = [float(x) for x in embedding[0]['embedding']]

                        FACES.objects.create(
                            PHOTO=photo,
                            FACE_LOCATION={
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h,
                                'landmarks': keypoints,
                                'face_image': f"faces/{face_filename}"
                            },
                            FACE_EMBEDDING=embedding_list,
                            DETECTION_CONFIDENCE=float(face_info['confidence'])
                        )
                        faces_processed += 1

                except Exception as face_error:
                    print(f"Error processing face: {str(face_error)}")
                    continue

            return JsonResponse({
                'message': f'Successfully processed {faces_processed} faces in image',
                'faces_detected': len(detected_faces),
                'faces_processed': faces_processed
            })

        except PHOTOS.DoesNotExist:
            return JsonResponse({'error': f'Photo record not found for ID: {record_id}'}, status=404)
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
def cluster_faces(request):
    if request.method == 'POST':
        try:
            faces = FACES.objects.filter(CLUSTER__isnull=True).values('RECORD_ID', 'FACE_EMBEDDING')
            if not faces:
                return JsonResponse({'message': 'No unclustered faces found.'})

            embeddings = [face['FACE_EMBEDDING'] for face in faces]
            record_ids = [face['RECORD_ID'] for face in faces]

            # Normalize embeddings
            embeddings_array = np.array(embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1)
            normalized_embeddings = embeddings_array / norms[:, np.newaxis]

            # More lenient DBSCAN parameters
            dbscan = DBSCAN(
                eps=0.45,              # Increased from 0.3 to 0.45 for more inclusive clustering
                min_samples=2,         # Keep minimum cluster size at 2
                metric='cosine',       # Using cosine similarity
                n_jobs=-1
            )
            
            cluster_labels = dbscan.fit_predict(normalized_embeddings)

            # Count unique clusters before processing
            unique_clusters = set(label for label in cluster_labels if label != -1)
            print(f"Initial number of clusters: {len(unique_clusters)}")

            # Reset existing clusters
            FACES.objects.filter(RECORD_ID__in=record_ids).update(CLUSTER=None)

            # Dictionary to store cluster centroids
            cluster_centroids = {}

            # Calculate centroids for each cluster
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = normalized_embeddings[cluster_mask]
                centroid = np.mean(cluster_embeddings, axis=0)
                cluster_centroids[cluster_id] = centroid

            # Merge similar clusters
            merged_labels = cluster_labels.copy()
            merge_threshold = 0.6  # Threshold for merging clusters

            for c1 in unique_clusters:
                for c2 in unique_clusters:
                    if c1 < c2:  # Only check each pair once
                        centroid1 = cluster_centroids[c1]
                        centroid2 = cluster_centroids[c2]
                        similarity = np.dot(centroid1, centroid2)
                        
                        if similarity > merge_threshold:
                            # Merge clusters by updating labels
                            merged_labels[merged_labels == c2] = c1

            # Create new clusters with merged labels
            for record_id, cluster_id in zip(record_ids, merged_labels):
                if cluster_id != -1:
                    cluster, created = FACE_CLUSTERS.objects.get_or_create(
                        NAME=f'Person {cluster_id + 1}'  # More friendly naming
                    )
                    FACES.objects.filter(RECORD_ID=record_id).update(CLUSTER=cluster)

            # Update cluster face counts and clean up empty clusters
            clusters = FACE_CLUSTERS.objects.all()
            for cluster in clusters:
                face_count = FACES.objects.filter(CLUSTER=cluster).count()
                if face_count == 0:
                    cluster.delete()
                else:
                    cluster.FACE_COUNT = face_count
                    cluster.save()

            final_cluster_count = FACE_CLUSTERS.objects.count()
            return JsonResponse({
                'message': 'Clustering completed successfully.',
                'clusters_created': final_cluster_count,
                'faces_processed': len(faces)
            })

        except Exception as e:
            print(f"Clustering error: {str(e)}")
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

            # Modified query to get all photos containing faces from this cluster
            if cluster_name:
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
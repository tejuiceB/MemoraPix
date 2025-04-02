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

            # Initialize MTCNN without parameters
            detector = MTCNN()

            # Load and process image
            img_array = cv2.imread(image_path)
            if img_array is None:
                return JsonResponse({'error': 'Could not read image file'}, status=400)

            # Resize large images to improve performance
            max_dimension = 1500
            height, width = img_array.shape[:2]
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / max(width, height)
                img_array = cv2.resize(img_array, None, fx=scale, fy=scale)

            # Convert BGR to RGB (MTCNN expects RGB)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            # Try different image preprocessing techniques
            detected_faces = []
            
            # Try original image
            faces = detector.detect_faces(img_array)
            if faces:
                detected_faces.extend(faces)
            
            # If no faces found, try with histogram equalization
            if not detected_faces:
                img_eq = cv2.equalizeHist(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
                img_eq = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
                faces = detector.detect_faces(img_eq)
                if faces:
                    detected_faces.extend(faces)

            # If still no faces, try with different scales
            if not detected_faces:
                scales = [1.0, 0.5, 1.5, 2.0]  # Added more scales
                for scale in scales:
                    scaled_img = cv2.resize(img_array, None, fx=scale, fy=scale)
                    faces = detector.detect_faces(scaled_img)
                    if faces:
                        # Adjust coordinates back to original scale
                        for face in faces:
                            if face['confidence'] >= 0.8:  # Reduced threshold from 0.9
                                face['box'] = [int(x/scale) for x in face['box']]
                                detected_faces.append(face)

            faces_processed = 0

            for face_info in detected_faces:
                try:
                    x, y, w, h = face_info['box']
                    
                    # Increase margin for better feature capture
                    margin_w = int(w * 0.3)  # Increased from 0.2
                    margin_h = int(h * 0.3)  # Increased from 0.2
                    x = max(0, x - margin_w)
                    y = max(0, y - margin_h)
                    w = min(img_array.shape[1] - x, w + 2 * margin_w)
                    h = min(img_array.shape[0] - y, h + 2 * margin_h)

                    # Extract and validate face region
                    if w < 30 or h < 30:  # Skip very small faces
                        continue
                        
                    face_img = img_array[y:y+h, x:x+w]
                    if face_img.size == 0:
                        continue

                    # Save face image with unique name
                    face_filename = f"face_{str(uuid.uuid4())[:8]}.jpg"
                    face_path = os.path.join(settings.MEDIA_ROOT, 'faces', face_filename)
                    os.makedirs(os.path.dirname(face_path), exist_ok=True)
                    cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

                    # Get face embedding
                    try:
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

                            # Save face to database
                            FACES.objects.create(
                                PHOTO=photo,
                                FACE_LOCATION=face_location,
                                FACE_EMBEDDING=embedding[0]['embedding'],
                                DETECTION_CONFIDENCE=face_info['confidence']
                            )
                            faces_processed += 1

                    except Exception as embed_error:
                        print(f"Error getting embedding: {str(embed_error)}")
                        continue

                except Exception as face_error:
                    print(f"Error processing face: {str(face_error)}")
                    continue

            # Run clustering if faces were found
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
    """Improved clustering with better parameters for small groups"""
    try:
        all_faces = FACES.objects.all()
        if not all_faces.exists():
            return JsonResponse({'message': 'No faces found.'}) if request else None

        # Prepare embeddings with proper normalization
        embeddings = []
        face_ids = []
        
        for face in all_faces:
            embedding = np.array(face.FACE_EMBEDDING)
            # Normalize each embedding vector
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            face_ids.append(face.RECORD_ID)

        embeddings_array = np.array(embeddings)

        # More lenient clustering parameters
        clustering = DBSCAN(
            eps=0.65,  # Increased from 0.55 to be more inclusive
            min_samples=1,  # Reduced from 2 to allow single-image clusters
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
                # Get faces for this cluster
                mask = labels == label
                cluster_face_ids = [face_ids[i] for i, is_member in enumerate(mask) if is_member]
                
                # Create cluster even for single faces
                cluster = FACE_CLUSTERS.objects.create(
                    NAME=f'Person_{clusters_created}',
                    FACE_COUNT=0
                )
                clusters_created += 1

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
            from concurrent.futures import ThreadPoolExecutor
            import threading
            
            # Get all photos
            unprocessed_photos = PHOTOS.objects.all()
            processed_count = threading.Value('i', 0)
            error_count = threading.Value('i', 0)
            
            def process_single_photo(photo):
                try:
                    # Load and process each image
                    image_path = os.path.join('media', photo.FILE_PATH)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        with error_count.get_lock():
                            error_count.value += 1
                        return

                    # Initialize MTCNN for this thread
                    detector = MTCNN(
                        min_face_size=20,
                        scale_factor=0.709,
                        steps_threshold=[0.6, 0.7, 0.7]
                    )

                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    faces = detector.detect_faces(image_rgb)

                    # Process each detected face
                    for face_info in faces:
                        if face_info['confidence'] < 0.8:  # Reduced threshold
                            continue

                        x, y, w, h = face_info['box']
                        face_img = image_rgb[y:y+h, x:x+w]
                        
                        # Save face image
                        face_filename = f"face_{str(uuid.uuid4())[:8]}.jpg"
                        face_path = os.path.join(settings.MEDIA_ROOT, 'faces', face_filename)
                        os.makedirs(os.path.dirname(face_path), exist_ok=True)
                        cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

                        # Get embedding
                        embedding = DeepFace.represent(
                            img_path=face_path,
                            model_name='VGG-Face',
                            enforce_detection=False,
                            detector_backend='skip'
                        )

                        if embedding:
                            face_location = {
                                'x': int(x),
                                'y': int(y),
                                'width': int(w),
                                'height': int(h),
                                'face_image': f"faces/{face_filename}"
                            }

                            FACES.objects.create(
                                PHOTO=photo,
                                FACE_LOCATION=face_location,
                                FACE_EMBEDDING=embedding[0]['embedding'],
                                DETECTION_CONFIDENCE=face_info['confidence']
                            )

                    with processed_count.get_lock():
                        processed_count.value += 1

                except Exception as e:
                    print(f"Error processing photo {photo.FILE_NAME}: {str(e)}")
                    with error_count.get_lock():
                        error_count.value += 1

            # Process photos in parallel
            max_workers = min(8, len(unprocessed_photos))  # Limit max threads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(process_single_photo, unprocessed_photos)

            # Run clustering with improved parameters
            cluster_faces(None)

            return JsonResponse({
                'message': f'Processed {processed_count.value} photos successfully. {error_count.value} errors occurred.',
                'processed_count': processed_count.value,
                'error_count': error_count.value
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
"""
URL configuration for FaceLensBackend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from system.views import (upload_image, process_image, cluster_faces, 
                        assign_cluster_name, search_images, process_all_photos, get_clusters)

urlpatterns = [
    path('upload/', upload_image, name='upload_image'),
    path('process/', process_image, name='process_image'),
    path('cluster/', cluster_faces, name='cluster_faces'),
    path('assign-name/', assign_cluster_name, name='assign_cluster_name'),
    path('search/', search_images, name='search_images'),
    path('process-all/', process_all_photos, name='process_all_photos'),
    path('clusters/', get_clusters, name='get_clusters'),
    path('media/<path:path>', serve, {'document_root': settings.MEDIA_ROOT}),  # Add this line
]

# Add this at the end of the file to serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

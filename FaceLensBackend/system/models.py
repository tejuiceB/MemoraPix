from django.db import models
from django.contrib.postgres.fields import ArrayField

class PHOTOS(models.Model):
    RECORD_ID = models.AutoField(primary_key=True)
    FILE_NAME = models.CharField(max_length=255)
    FILE_PATH = models.CharField(max_length=1000)
    UPLOAD_DATE = models.DateTimeField(auto_now_add=True)
    LOCATION = models.CharField(max_length=255, null=True, blank=True)
    METADATA = models.JSONField(null=True, blank=True)
    CREATED_AT = models.DateTimeField(auto_now_add=True)
    UPDATED_AT = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'PHOTOS'

class FACE_CLUSTERS(models.Model):
    RECORD_ID = models.AutoField(primary_key=True)
    NAME = models.CharField(max_length=255)
    FACE_COUNT = models.IntegerField(default=0)
    CREATED_AT = models.DateTimeField(auto_now_add=True)
    UPDATED_AT = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'FACE_CLUSTERS'

class FACES(models.Model):
    RECORD_ID = models.AutoField(primary_key=True)
    PHOTO = models.ForeignKey('PHOTOS', on_delete=models.CASCADE)
    FACE_LOCATION = models.JSONField()
    FACE_EMBEDDING = ArrayField(models.FloatField())
    CLUSTER = models.ForeignKey('FACE_CLUSTERS', on_delete=models.SET_NULL, null=True)
    DETECTION_CONFIDENCE = models.FloatField()
    CREATED_AT = models.DateTimeField(auto_now_add=True)
    UPDATED_AT = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'FACES'

class USERS(models.Model):
    RECORD_ID = models.AutoField(primary_key=True)
    USERNAME = models.CharField(max_length=255, unique=True)
    EMAIL = models.EmailField(unique=True)
    PASSWORD_HASH = models.CharField(max_length=255)
    CREATED_AT = models.DateTimeField(auto_now_add=True)
    UPDATED_AT = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'USERS'

class PHOTO_LABELS(models.Model):
    RECORD_ID = models.AutoField(primary_key=True)
    PHOTO = models.ForeignKey('PHOTOS', on_delete=models.CASCADE)
    LABEL_TEXT = models.CharField(max_length=255)
    CREATED_AT = models.DateTimeField(auto_now_add=True)
    UPDATED_AT = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'PHOTO_LABELS'

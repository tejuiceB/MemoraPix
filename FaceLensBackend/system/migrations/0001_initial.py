# Generated by Django 5.1.7 on 2025-04-01 06:19

import django.contrib.postgres.fields
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='FACE_CLUSTERS',
            fields=[
                ('RECORD_ID', models.AutoField(primary_key=True, serialize=False)),
                ('NAME', models.CharField(max_length=255)),
                ('FACE_COUNT', models.IntegerField(default=0)),
                ('CREATED_AT', models.DateTimeField(auto_now_add=True)),
                ('UPDATED_AT', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'FACE_CLUSTERS',
            },
        ),
        migrations.CreateModel(
            name='PHOTOS',
            fields=[
                ('RECORD_ID', models.AutoField(primary_key=True, serialize=False)),
                ('FILE_NAME', models.CharField(max_length=255)),
                ('FILE_PATH', models.CharField(max_length=1000)),
                ('UPLOAD_DATE', models.DateTimeField(auto_now_add=True)),
                ('LOCATION', models.CharField(blank=True, max_length=255, null=True)),
                ('METADATA', models.JSONField(blank=True, null=True)),
                ('CREATED_AT', models.DateTimeField(auto_now_add=True)),
                ('UPDATED_AT', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'PHOTOS',
            },
        ),
        migrations.CreateModel(
            name='USERS',
            fields=[
                ('RECORD_ID', models.AutoField(primary_key=True, serialize=False)),
                ('USERNAME', models.CharField(max_length=255, unique=True)),
                ('EMAIL', models.EmailField(max_length=254, unique=True)),
                ('PASSWORD_HASH', models.CharField(max_length=255)),
                ('CREATED_AT', models.DateTimeField(auto_now_add=True)),
                ('UPDATED_AT', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'USERS',
            },
        ),
        migrations.CreateModel(
            name='PHOTO_LABELS',
            fields=[
                ('RECORD_ID', models.AutoField(primary_key=True, serialize=False)),
                ('LABEL_TEXT', models.CharField(max_length=255)),
                ('CREATED_AT', models.DateTimeField(auto_now_add=True)),
                ('UPDATED_AT', models.DateTimeField(auto_now=True)),
                ('PHOTO', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='system.photos')),
            ],
            options={
                'db_table': 'PHOTO_LABELS',
            },
        ),
        migrations.CreateModel(
            name='FACES',
            fields=[
                ('RECORD_ID', models.AutoField(primary_key=True, serialize=False)),
                ('FACE_LOCATION', models.JSONField()),
                ('FACE_EMBEDDING', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('DETECTION_CONFIDENCE', models.FloatField()),
                ('CREATED_AT', models.DateTimeField(auto_now_add=True)),
                ('UPDATED_AT', models.DateTimeField(auto_now=True)),
                ('CLUSTER', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='system.face_clusters')),
                ('PHOTO', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='system.photos')),
            ],
            options={
                'db_table': 'FACES',
            },
        ),
    ]

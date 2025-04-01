import React, { useState } from 'react';
import { uploadImage, processImage } from '../api';

function ImageUpload() {
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [message, setMessage] = useState('');
    const [isError, setIsError] = useState(false);

    const handleFileChange = (e) => {
        setFiles(Array.from(e.target.files));
    };

    const handleUpload = async () => {
        if (files.length === 0) {
            setMessage('Please select files to upload.');
            setIsError(true);
            return;
        }

        setUploading(true);
        setProgress(0);
        let successCount = 0;

        try {
            for (let i = 0; i < files.length; i++) {
                const formData = new FormData();
                formData.append('image', files[i]);

                const uploadResponse = await uploadImage(formData);
                
                if (uploadResponse.status === 200 && uploadResponse.data.record_id) {
                    try {
                        // Process the uploaded image
                        await processImage(uploadResponse.data.record_id);
                        successCount++;
                    } catch (processError) {
                        console.error('Error processing image:', processError);
                    }
                }

                setProgress(Math.round(((i + 1) / files.length) * 100));
            }

            setMessage(`Successfully uploaded and processed ${successCount} of ${files.length} images.`);
            setIsError(false);
            setFiles([]);
            document.querySelector('input[type="file"]').value = '';
            
        } catch (error) {
            console.error('Error:', error);
            setMessage('Error processing images. Please try again.');
            setIsError(true);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="image-upload">
            <h2>Upload Images</h2>
            <input 
                type="file" 
                onChange={handleFileChange} 
                multiple 
                accept="image/*"
                disabled={uploading}
            />
            <button 
                onClick={handleUpload} 
                disabled={uploading || files.length === 0}
            >
                {uploading ? `Uploading... ${progress}%` : 'Upload Images'}
            </button>
            {uploading && (
                <div className="progress-bar">
                    <div 
                        className="progress" 
                        style={{ width: `${progress}%` }}
                    ></div>
                </div>
            )}
            <p style={{ color: isError ? 'red' : 'green' }}>
                {message}
            </p>
            {files.length > 0 && (
                <div className="selected-files">
                    <h3>Selected Files:</h3>
                    <ul>
                        {files.map((file, index) => (
                            <li key={index}>{file.name}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

export default ImageUpload;

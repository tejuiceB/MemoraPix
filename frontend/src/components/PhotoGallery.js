import React, { useState, useEffect } from 'react';
import axios from 'axios';

function PhotoGallery() {
    const [clusters, setClusters] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchClusters = async () => {
        try {
            console.log('Fetching clusters...'); // Debug log
            const clustersResponse = await axios.get('http://127.0.0.1:8000/clusters/', {
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                }
            });

            console.log('Clusters response:', clustersResponse.data); // Debug log

            if (!clustersResponse.data.clusters) {
                throw new Error('No clusters data received');
            }

            const clustersWithPhotos = await Promise.all(
                clustersResponse.data.clusters.map(async (cluster) => {
                    console.log(`Fetching photos for cluster: ${cluster.NAME}`); // Debug log
                    try {
                        const photosResponse = await axios.get(`http://127.0.0.1:8000/search/`, {
                            params: {
                                cluster_name: cluster.NAME
                            }
                        });
                        console.log(`Photos for cluster ${cluster.NAME}:`, photosResponse.data); // Debug log
                        return {
                            ...cluster,
                            photos: photosResponse.data.photos || []
                        };
                    } catch (photoError) {
                        console.error(`Error fetching photos for cluster ${cluster.NAME}:`, photoError);
                        return {
                            ...cluster,
                            photos: []
                        };
                    }
                })
            );

            setClusters(clustersWithPhotos);
            setError(null);
        } catch (err) {
            console.error('Error in fetchClusters:', err); // Debug log
            setError(`Error fetching data: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchClusters();
    }, []);

    if (loading) return <div>Loading clusters and photos...</div>;
    if (error) return (
        <div>
            <p style={{ color: 'red' }}>{error}</p>
            <button onClick={fetchClusters}>Retry</button>
        </div>
    );

    return (
        <div className="photo-gallery">
            <h2>Photo Clusters ({clusters.length} clusters found)</h2>
            {clusters.length === 0 ? (
                <p>No clusters found. Try processing some images first.</p>
            ) : (
                clusters.map((cluster) => (
                    <div key={cluster.RECORD_ID} className="cluster-section">
                        <h3>{cluster.NAME} ({cluster.photos.length} photos)</h3>
                        <div className="photo-grid">
                            {cluster.photos.length === 0 ? (
                                <p>No photos in this cluster</p>
                            ) : (
                                cluster.photos.map((photo) => (
                                    <div key={photo.RECORD_ID} className="photo-item">
                                        <img 
                                            src={`http://127.0.0.1:8000/media/${photo.FILE_PATH}`}
                                            alt={photo.FILE_NAME}
                                            onError={(e) => {
                                                console.error(`Error loading image: ${photo.FILE_PATH}`);
                                                e.target.src = 'https://via.placeholder.com/150?text=Image+Not+Found';
                                            }}
                                            style={{
                                                width: '200px',
                                                height: '200px',
                                                objectFit: 'cover'
                                            }}
                                        />
                                        <p>{photo.FILE_NAME}</p>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                ))
            )}
        </div>
    );
}

export default PhotoGallery;

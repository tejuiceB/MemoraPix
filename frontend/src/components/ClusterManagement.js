import React, { useState } from 'react';
import { clusterFaces, assignClusterName } from '../api';

function ClusterManagement() {
  const [message, setMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [clusterId, setClusterId] = useState('');
  const [name, setName] = useState('');

  const handleCluster = async () => {
    try {
      setIsProcessing(true);
      setMessage('Running face clustering...');
      
      const response = await clusterFaces();
      if (response.data.message === 'No unclustered faces found.') {
        setMessage('No new faces to cluster. Try uploading and processing images first.');
      } else {
        setMessage('Clustering completed successfully! You can now assign names to clusters.');
      }
    } catch (error) {
      console.error('Clustering error:', error);
      setMessage('Error during clustering. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAssignName = async () => {
    try {
      await assignClusterName(clusterId, name);
      setMessage(`Cluster ${clusterId} renamed to ${name}.`);
    } catch (error) {
      setMessage('Error assigning name to cluster.');
    }
  };

  return (
    <div>
      <h2>Face Clustering</h2>
      <button 
        onClick={handleCluster} 
        disabled={isProcessing}
      >
        {isProcessing ? 'Processing...' : 'Run Face Clustering'}
      </button>
      <div>
        <input
          type="text"
          placeholder="Cluster ID"
          value={clusterId}
          onChange={(e) => setClusterId(e.target.value)}
        />
        <input
          type="text"
          placeholder="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button onClick={handleAssignName}>Assign Name</button>
      </div>
      <p>{message}</p>
    </div>
  );
}

export default ClusterManagement;

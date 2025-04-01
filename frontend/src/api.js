import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

export const uploadImage = (formData) => {
  return axios.post(`${API_BASE_URL}/upload/`, formData, {
    headers: { 
      'Content-Type': 'multipart/form-data'
    },
    withCredentials: true
  });
};

export const processImage = (recordId) => {
  const formData = new FormData();
  formData.append('record_id', recordId);
  
  return axios.post(`${API_BASE_URL}/process/`, formData);
};

export const clusterFaces = () => {
  return axios.post(`${API_BASE_URL}/cluster/`);
};

export const assignClusterName = (clusterId, name) => {
  return axios.post(`${API_BASE_URL}/assign-name/`, { cluster_id: clusterId, name });
};

export const searchImages = (params) => {
  return axios.get(`${API_BASE_URL}/search/`, { params });
};

export const getClusters = () => {
  return axios.get(`${API_BASE_URL}/clusters/`);
};

export const getPhotosForCluster = (clusterName) => {
  return axios.get(`${API_BASE_URL}/search/`, {
    params: { cluster_name: clusterName }
  });
};

// Add a new function to check if an image exists
export const checkImageExists = async (imagePath) => {
  try {
    const response = await axios.head(`${API_BASE_URL}/${imagePath}`);
    return response.status === 200;
  } catch (error) {
    return false;
  }
};

import React, { useState } from 'react';
import { searchImages } from '../api';

function Search() {
  const [params, setParams] = useState({});
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    try {
      const response = await searchImages(params);
      setResults(response.data.photos);
    } catch (error) {
      console.error('Error searching images:', error);
    }
  };

  const handleChange = (e) => {
    setParams({ ...params, [e.target.name]: e.target.value });
  };

  return (
    <div>
      <h2>Search Images</h2>
      <input name="cluster_name" placeholder="Cluster Name" onChange={handleChange} />
      <input name="face_id" placeholder="Face ID" onChange={handleChange} />
      <input name="start_date" type="date" onChange={handleChange} />
      <input name="end_date" type="date" onChange={handleChange} />
      <button onClick={handleSearch}>Search</button>
      <div>
        {results.map((photo) => (
          <div key={photo.RECORD_ID}>
            <img 
              src={`http://127.0.0.1:8000/media/${photo.FILE_NAME}`}
              alt={photo.FILE_NAME}
              onError={(e) => {
                e.target.src = 'https://via.placeholder.com/150?text=Image+Not+Found';
              }}
              style={{
                width: '200px',
                height: '200px',
                objectFit: 'cover',
                margin: '5px'
              }}
            />
            <p>{photo.FILE_NAME}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Search;

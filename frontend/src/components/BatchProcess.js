import React, { useState } from 'react';
import axios from 'axios';

function BatchProcess() {
    const [isProcessing, setIsProcessing] = useState(false);
    const [message, setMessage] = useState('');

    const handleBatchProcess = async () => {
        setIsProcessing(true);
        setMessage('Processing all photos... This may take a while.');

        try {
            const response = await axios.post('http://127.0.0.1:8000/process-all/');
            setMessage(response.data.message);
        } catch (error) {
            setMessage('Error processing photos: ' + error.message);
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div>
            <h2>Batch Process Photos</h2>
            <button 
                onClick={handleBatchProcess} 
                disabled={isProcessing}
            >
                {isProcessing ? 'Processing...' : 'Process All Photos'}
            </button>
            <p>{message}</p>
        </div>
    );
}

export default BatchProcess;

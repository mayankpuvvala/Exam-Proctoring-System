import React, { useState } from 'react';
import { Button, Alert, Spinner, Card } from 'react-bootstrap';

const FaceTurn = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const handleStartCamera = async () => {
        setLoading(true);
        setError('');
        setSuccess('');

        try {
            const response = fetch('http://localhost:8000/start_camera', {
                method: 'GET',
            });

            const data = await response.text();
            setSuccess(data);
        } catch (err) {
            console.error('Error starting camera:', err);
            setError('Failed to start the camera. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleStopCamera = async () => {
        setLoading(true);
        setError('');
        setSuccess('');

        try {
            const response = await fetch('http://localhost:8000/stop_camera', {
                method: 'GET',
            });

            const data = await response.text();
            setSuccess(data);
        } catch (err) {
            console.error('Error stopping camera:', err);
            setError('Failed to stop the camera. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="d-flex justify-content-center align-items-center vh-100 bg-light">
            <Card style={{ width: '300px' }} className="shadow">
                <Card.Body>
                    <h3 className="text-center mb-4">Face Turn Detection</h3>

                    {error && <Alert variant="danger">{error}</Alert>}
                    {success && <Alert variant="success">{success}</Alert>}

                    <div className="text-center">
                        <Button variant="primary" className="me-2" onClick={handleStartCamera} disabled={loading}>
                            {loading ? <Spinner animation="border" size="sm" /> : 'Start Camera'}
                        </Button>
                        <Button variant="danger" onClick={handleStopCamera} disabled={loading}>
                            {loading ? <Spinner animation="border" size="sm" /> : 'Stop Camera'}
                        </Button>
                    </div>
                </Card.Body>
            </Card>
        </div>
    );
};

export default FaceTurn;

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Button, Alert, Spinner } from 'react-bootstrap';

const Verification = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const navigate = useNavigate();

    const verifyUser = async () => {
        setLoading(true);
        setError('');
        setSuccess('');

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const video = document.createElement('video');
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageDataURL = canvas.toDataURL('image/jpeg');

            stream.getTracks().forEach(track => track.stop());

            const response = await fetch('/verify_user', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ photo: imageDataURL }),
            });

            const data = await response.json();

            if (data.error) {
                setError(data.error);
            } else {
                setSuccess(data.message);
                setTimeout(() => {
                    navigate('/exam');
                }, 2000);
            }
        } catch (err) {
            console.error('Error during verification:', err);
            setError('Verification failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="d-flex justify-content-center align-items-center vh-100 bg-light">
            <Card style={{ width: '400px' }} className="shadow">
                <Card.Body className="text-center">
                    <h3 className="mb-4">User Verification</h3>

                    {error && <Alert variant="danger">{error}</Alert>}
                    {success && <Alert variant="success">{success}</Alert>}

                    <Button variant="primary" onClick={verifyUser} disabled={loading}>
                        {loading ? <Spinner animation="border" size="sm" /> : 'Start Verification'}
                    </Button>
                </Card.Body>
            </Card>
        </div>
    );
};

export default Verification;

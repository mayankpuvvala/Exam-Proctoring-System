import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Navbar, Nav, Card, Form, Button, Alert, Spinner } from 'react-bootstrap';
import '../styles/Registration.css';

const Registration = () => {
    const [username, setUsername] = useState('');
    const [capturedImage, setCapturedImage] = useState(null);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const capturePhoto = async () => {
        setError('');
        setSuccess('');
        setLoading(true);

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

            setCapturedImage(imageDataURL);
            setSuccess('Photo captured successfully!');
            stream.getTracks().forEach(track => track.stop());
        } catch (err) {
            console.error('Error capturing photo:', err);
            setError('Failed to capture photo. Please allow camera access.');
        } finally {
            setLoading(false);
        }
    };

    const registerUser = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setLoading(true);

        if (!username || !capturedImage) {
            setError('Please enter your name and capture a photo.');
            setLoading(false);
            return;
        }

        try {
            const response = await fetch('/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name: username, photo: capturedImage }),
            });

            const data = await response.json();

            if (data.error) {
                setError(data.error);
            } else {
                setSuccess(data.message);
                setTimeout(() => {
                    navigate('/verification');
                }, 2000);
            }
        } catch (err) {
            console.error('Error registering user:', err);
            setError('An unexpected error occurred. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <Navbar bg="dark" variant="dark" expand="lg" className="mb-4">
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="ml-auto">
                        <Nav.Link href="/">Home</Nav.Link>
                        <Nav.Link href="/faceturn">Face Turn</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>

            <div className="registration-container">
                <Card className="registration-card">
                    <Card.Body>
                        <h3 className="text-center mb-4">User Registration</h3>

                        {error && <Alert variant="danger">{error}</Alert>}
                        {success && <Alert variant="success">{success}</Alert>}

                        <Form onSubmit={registerUser}>
                            <Form.Group controlId="username" className="mb-3">
                                <Form.Label>Full Name</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="Enter your full name"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                />
                            </Form.Group>

                            <Form.Group controlId="photo" className="mb-3 text-center">
                                {capturedImage ? (
                                    <img
                                        src={capturedImage}
                                        alt="Captured"
                                        className="img-fluid rounded mb-3 registration-image"
                                    />
                                ) : (
                                    <div className="mb-3 text-muted">No photo captured</div>
                                )}
                                <Button variant="primary" onClick={capturePhoto} disabled={loading}>
                                    {loading ? <Spinner animation="border" size="sm" /> : 'Capture Photo'}
                                </Button>
                            </Form.Group>

                            <Button variant="success" type="submit" className="w-100" disabled={loading}>
                                {loading ? <Spinner animation="border" size="sm" /> : 'Register'}
                            </Button>
                        </Form>
                    </Card.Body>
                </Card>
            </div>
        </div>
    );
};

export default Registration;

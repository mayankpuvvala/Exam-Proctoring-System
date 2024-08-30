import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Registration from './pages/Registration';
import Verification from './pages/Verification';
import Exam from './pages/Exam';
import Status from './pages/Status';
import { Navbar, Container } from 'react-bootstrap';
import FaceTurn from './pages/FaceTurn';

function App() {
    return (
        <Router>
            <Navbar bg="dark" variant="dark">
                <Container>
                    <Navbar.Brand href="/">Exam Proctoring System</Navbar.Brand>
                </Container>
            </Navbar>
            <Routes>
                <Route path="/" element={<Registration />} />
                <Route path="/faceturn" element={<FaceTurn />} />
                <Route path="/verification" element={<Verification />} />
                <Route path="/exam" element={<Exam />} />
                <Route path="/status" element={<Status />} />
            </Routes>
        </Router>
    );
}

export default App;

const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');

const app = express();

const cors = require('cors');
app.use(cors());

const PORT = 5000;

// Create storage engine for multer
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const dir = './database';
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        cb(null, dir);
    },
    filename: (req, file, cb) => {
        const username = req.body.username;
        cb(null, `${username}.jpg`);
    }
});

const upload = multer({ storage: storage });

// Middleware to parse JSON bodies (not needed for FormData)
app.use(express.json());

// Endpoint to handle registration
app.post('/register', upload.single('photo'), (req, res) => {
    console.log('Received request:', req.body);
    console.log('File:', req.file);

    const username = req.body.username;
    if (!username) {
        return res.status(400).json({ error: 'Username is required' });
    }

    if (!req.file) {
        return res.status(400).json({ error: 'Photo is required' });
    }

    return res.json({ message: 'Registration successful!' });
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

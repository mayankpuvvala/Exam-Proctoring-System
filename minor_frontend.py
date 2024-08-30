<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exam Proctoring System</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <link rel="stylesheet" type="text/css" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <h1>Monitoring Portal</h1>
        <div id="exam">
            <h2>Exam in Progress</h2>
            <img id="video" src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>
    </div>
        
    <div id="status">
        <div class="status-item">
            <span class="status-label">Lighting:</span>
            <span id="lighting-status">Loading...</span>
        </div>
        <div class="status-item">
            <span class="status-label">Blur:</span>
            <span id="blur-status">Loading...</span>
        </div>
        <div class="status-item">
            <span class="status-label">Face Detection:</span>
            <span id="face-status">Loading...</span>
        </div>
        <div class="status-item">
            <span class="status-label">Tab Switches:</span>
            <span id="tab-switches">0</span>
        </div>
        <div class="status-item">
            <span class="status-label">Inactive Time:</span>
            <span id="inactive-time">0 seconds</span>
        </div>
    </div>
    <script>
        let tabSwitches = 0;
        let inactiveTime = 0;
        let inactiveInterval;
        let isActive = true;
    
        function tabSwitch() {
    $.ajax({
        url: '/tab_switch',
        type: 'GET',
        success: function(data) {
            tabSwitches = data.tab_switches;
            $('#tab-switches').text(tabSwitches);
            if (tabSwitches >= 5) {
                alert("Exam stopped due to excessive tab switching.");
                window.location.href = '/stop_camera';  // Redirect and stop camera
            }
        },
        error: function(xhr, status, error) {
            if (xhr.status === 403) {
                alert("Exam has been stopped due to excessive tab switching.");
                window.location.href = '/stop_camera';
            } else {
                console.error('Error recording tab switch:', error);
            }
        }
    });
}

    
        function startInactiveTimer() {
            isActive = false;
            inactiveInterval = setInterval(() => {
                inactiveTime++;
                $('#inactive-time').text(inactiveTime + ' seconds');
            }, 1000);
        }
    
        function stopInactiveTimer() {
            isActive = true;
            clearInterval(inactiveInterval);
        }
    
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                tabSwitch();
                startInactiveTimer();
            } else {
                stopInactiveTimer();
            }
        });
        function updateStatus(data) {
        console.log("Updating status with data:", data);
        $('#lighting-status').text(data.lighting !== "N/A" ? data.lighting : "Good");
        $('#blur-status').text(data.blur !== "N/A" ? data.blur : "Visible");
        $('#face-status').text(data.face_detected ? 'Face Detected' : 'Face Detected');
    }

        setInterval(() => {
            fetch('/status')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    updateStatus(data);
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                    $('#lighting-status').text('Error');
                    $('#blur-status').text('Error');
                    $('#face-status').text('Error');
                });
        }, 5000);
    </script>
    
</body>
</html>

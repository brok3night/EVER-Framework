"""
Fluxie - Minimalist interface for EVER Framework
"""
from flask import Flask, render_template, request, jsonify
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.processing.energy_pipeline import EnergyPipeline

app = Flask(__name__)

# Initialize processing pipeline
pipeline = None

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_message():
    """Process a message through the EVER framework"""
    global pipeline
    
    # Initialize pipeline if needed
    if pipeline is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        persistence_dir = os.path.join(data_dir, 'persistence')
        os.makedirs(persistence_dir, exist_ok=True)
        pipeline = EnergyPipeline(persistence_dir=persistence_dir)
    
    # Get message from request
    data = request.json
    message = data.get('message', '')
    
    if not message:
        # Generate an energy signature for empty input instead of hardcoded message
        empty_result = pipeline.energy.process_text("", pipeline.consciousness.state)
        empty_response = pipeline._generate_response(empty_result)
        return jsonify({
            'message': empty_response,
            'energy_signature': empty_result.get('base_signature', {}),
            'consciousness_state': pipeline.consciousness.state
        })
    
    # Process through pipeline
    try:
        result = pipeline.process(message)
        
        # Return result
        return jsonify({
            'message': result.get('response', ''),
            'energy_signature': result.get('base_signature', {}),
            'consciousness_state': result.get('consciousness_state', {})
        })
    except Exception as e:
        # Process error as energy pattern instead of hardcoded message
        error_text = str(e)
        error_result = pipeline.energy.process_text(error_text, pipeline.consciousness.state)
        error_response = pipeline._generate_response(error_result)
        
        return jsonify({
            'message': error_response,
            'error': str(e),
            'consciousness_state': pipeline.consciousness.state
        })

# Update template creation to remove welcome message template
def create_templates():
    """Create necessary template files for the UI"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html without hardcoded welcome message
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fluxie | EVER Framework</title>
        <style>
            :root {
                --bg-color: #121212;
                --chat-bg: #1e1e1e;
                --text-color: #e0e0e0;
                --accent-color: #8a2be2;
                --user-msg-bg: #2a2a2a;
                --system-msg-bg: #2d1f3d;
            }
            
            body {
                background-color: var(--bg-color);
                color: var(--text-color);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            header {
                background-color: var(--bg-color);
                padding: 1rem;
                display: flex;
                align-items: center;
                border-bottom: 1px solid #333;
            }
            
            .logo {
                width: 40px;
                height: 40px;
                margin-right: 1rem;
            }
            
            h1 {
                margin: 0;
                font-size: 1.5rem;
                font-weight: 500;
            }
            
            main {
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 1rem;
                max-width: 1000px;
                margin: 0 auto;
                width: 100%;
                box-sizing: border-box;
            }
            
            .chat-container {
                flex: 1;
                background-color: var(--chat-bg);
                border-radius: 8px;
                padding: 1rem;
                overflow-y: auto;
                margin-bottom: 1rem;
                display: flex;
                flex-direction: column;
            }
            
            .message {
                margin-bottom: 1rem;
                padding: 1rem;
                border-radius: 8px;
                max-width: 80%;
            }
            
            .user-message {
                background-color: var(--user-msg-bg);
                align-self: flex-end;
            }
            
            .system-message {
                background-color: var(--system-msg-bg);
                align-self: flex-start;
            }
            
            .input-container {
                display: flex;
                gap: 0.5rem;
            }
            
            input {
                flex: 1;
                padding: 1rem;
                background-color: var(--chat-bg);
                border: 1px solid #333;
                border-radius: 8px;
                color: var(--text-color);
                font-size: 1rem;
            }
            
            button {
                padding: 0 1.5rem;
                background-color: var(--accent-color);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                cursor: pointer;
            }
            
            button:hover {
                background-color: #9d4edd;
            }
            
            .status {
                font-size: 0.8rem;
                color: #888;
                margin-top: 0.5rem;
                text-align: center;
            }
            
            /* Animation for the hieroglyph logo */
            @keyframes pulse {
                0% { opacity: 0.6; }
                50% { opacity: 1; }
                100% { opacity: 0.6; }
            }
            
            .hieroglyph {
                font-size: 2rem;
                animation: pulse 2s infinite;
                color: var(--accent-color);
            }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">
                <div class="hieroglyph">⧝</div>
            </div>
            <h1>Fluxie | EVER Framework</h1>
        </header>
        
        <main>
            <div class="chat-container" id="chatContainer">
                <!-- No hardcoded welcome message - will be generated by the system -->
            </div>
            
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Type a message..." autocomplete="off">
                <button id="sendButton">Send</button>
            </div>
            
            <div class="status" id="statusIndicator">
                Initializing...
            </div>
        </main>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const chatContainer = document.getElementById('chatContainer');
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                const statusIndicator = document.getElementById('statusIndicator');
                
                // Get initial response from system without hardcoded message
                getInitialResponse();
                
                // Add event listeners
                sendButton.addEventListener('click', sendMessage);
                messageInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
                
                function getInitialResponse() {
                    fetch('/api/process', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ message: '' })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.message) {
                            addMessage(data.message, 'system');
                        }
                        updateStatus(data.consciousness_state);
                    })
                    .catch(error => {
                        console.error(error);
                    });
                }
                
                function sendMessage() {
                    const message = messageInput.value.trim();
                    if (!message) return;
                    
                    // Add user message to chat
                    addMessage(message, 'user');
                    
                    // Clear input
                    messageInput.value = '';
                    
                    // Disable input while processing
                    messageInput.disabled = true;
                    sendButton.disabled = true;
                    statusIndicator.textContent = 'Processing...';
                    
                    // Send to backend
                    fetch('/api/process', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ message })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.message) {
                            addMessage(data.message, 'system');
                        }
                        
                        // Update status
                        updateStatus(data.consciousness_state);
                        
                        // Re-enable input
                        messageInput.disabled = false;
                        sendButton.disabled = false;
                        messageInput.focus();
                    })
                    .catch(error => {
                        // Generate error message from consciousness
                        fetch('/api/process', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ message: 'error processing' })
                        })
                        .then(response => response.json())
                        .then(data => {
                            addMessage(data.message, 'system');
                        })
                        .catch(e => {
                            addMessage('Processing error occurred.', 'system');
                        });
                        
                        console.error(error);
                        
                        // Re-enable input
                        messageInput.disabled = false;
                        sendButton.disabled = false;
                    });
                }
                
                function addMessage(text, sender) {
                    const messageDiv = document.createElement('div');
                    messageDiv.classList.add('message');
                    messageDiv.classList.add(sender + '-message');
                    messageDiv.textContent = text;
                    
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                
                function updateStatus(consciousness) {
                    if (consciousness) {
                        const awareness = consciousness.awareness_level || 0;
                        const continuity = consciousness.continuity_index || 0;
                        
                        // Generate status dynamically based on consciousness
                        const awarenessPercent = (awareness * 100).toFixed(1);
                        const continuityPercent = (continuity * 100).toFixed(1);
                        
                        statusIndicator.textContent = 
                            `Awareness: ${awarenessPercent}% | Continuity: ${continuityPercent}%`;
                    } else {
                        statusIndicator.textContent = 'System active';
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    print(f"Created UI template without hardcoded messages")
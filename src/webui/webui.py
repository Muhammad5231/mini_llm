def get_webui_html() -> str:
    """Returns responsive Single Page Chat Web UI HTML/JS string."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini LLM Version 4 Web UI</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: #1e1e1e; border-radius: 10px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        h2 { text-align: center; color: #4caf50; margin-top: 0; }
        #chat-box { height: 400px; overflow-y: auto; border: 1px solid #333; padding: 15px; border-radius: 8px; background: #181818; margin-bottom: 15px; }
        .message { margin-bottom: 15px; }
        .user { color: #2196f3; font-weight: bold; }
        .assistant { color: #4caf50; font-weight: bold; }
        .text { background: #252525; padding: 10px; border-radius: 6px; margin-top: 5px; display: inline-block; max-width: 90%; }
        .input-group { display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 12px; border: 1px solid #444; border-radius: 6px; background: #2a2a2a; color: #fff; font-size: 16px; }
        button { padding: 12px 24px; border: none; background: #4caf50; color: white; border-radius: 6px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🤖 Mini LLM Version 4 Web Interface</h2>
        <div id="chat-box"></div>
        <div class="input-group">
            <input type="text" id="prompt-input" placeholder="Type your message here..." onkeydown="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('prompt-input');
            const prompt = input.value.trim();
            if (!prompt) return;

            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="message"><span class="user">User:</span><div class="text">${prompt}</div></div>`;
            input.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await response.json();
                const reply = data.response || data.error;
                chatBox.innerHTML += `<div class="message"><span class="assistant">MiniLLM:</span><div class="text">${reply}</div></div>`;
            } catch (err) {
                chatBox.innerHTML += `<div class="message"><span class="assistant">Error:</span><div class="text">Failed to connect to API</div></div>`;
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>"""
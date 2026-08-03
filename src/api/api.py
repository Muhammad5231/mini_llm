import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from src.generate import generate_stream
from src.model import MiniLLM
from src.tokenizer import BPETokenizer

class MiniLLMRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler providing REST API endpoints (/api/chat, /api/health)."""
    
    model: MiniLLM = None
    tokenizer: BPETokenizer = None
    config = None

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path == "/api/health":
            self._set_headers(200)
            summary = self.model.get_model_summary()
            self.wfile.write(json.dumps({"status": "healthy", "summary": summary}).encode("utf-8"))
        elif self.path == "/" or self.path == "/chat":
            # Serve Web UI HTML
            self._set_headers(200, "text/html")
            from src.webui.webui import get_webui_html
            self.wfile.write(get_webui_html().encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                body = json.loads(post_data.decode("utf-8"))
                prompt = body.get("prompt", "").strip()

                if not prompt:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "Prompt cannot be empty"}).encode("utf-8"))
                    return

                formatted_prompt = f"User: {prompt}\nAssistant:"
                response_text = generate_stream(self.model, self.tokenizer, formatted_prompt, self.config)

                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "prompt": prompt,
                    "response": response_text
                }).encode("utf-8"))

            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def start_api_server(model: MiniLLM, tokenizer: BPETokenizer, config):
    """Launches lightweight local HTTP REST API server."""
    MiniLLMRequestHandler.model = model
    MiniLLMRequestHandler.tokenizer = tokenizer
    MiniLLMRequestHandler.config = config

    server_address = (config.api_host, config.api_port)
    httpd = HTTPServer(server_address, MiniLLMRequestHandler)
    print(f"\n========================================================")
    print(f" REST API & Web UI running at: http://{config.api_host}:{config.api_port}")
    print(f" Endpoints: GET /api/health | POST /api/chat | GET /chat")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping API server.")
        httpd.server_close()
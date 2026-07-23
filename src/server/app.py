import os
import sys
import json
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

# Add project root directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)

from src.preprocessor.image_processor import preprocess_image
from src.pipeline.context_builder import KokodivContextBuilder
from src.rag.retriever import CocoaRAGRetriever

ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv3_cocoa.onnx")
DB_PATH = os.path.join(BASE_DIR, "data", "kokodiv.db")
LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8080/completion")

classifier = None
if os.path.exists(ONNX_MODEL_PATH):
    try:
        from src.classifier.onnx_classifier import CocoaONNXClassifier
        classifier = CocoaONNXClassifier(ONNX_MODEL_PATH)
        print(f"[✓] ONNX Cocoa Classifier loaded from: {ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"[!] Warning loading ONNX classifier: {e}")

retriever = CocoaRAGRetriever(DB_PATH)
context_builder = KokodivContextBuilder(DB_PATH)

class KokodivRequestHandler(BaseHTTPRequestHandler):
    """
    Zero-dependency HTTP Conversational Application Server for Kokodiv.
    """
    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200, "text/plain")

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        parsed_path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        if parsed_path == "/":
            ui_path = os.path.join(BASE_DIR, "src", "ui", "index.html")
            if os.path.exists(ui_path):
                self._set_headers(200, "text/html; charset=utf-8")
                with open(ui_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers(404, "text/plain")
                self.wfile.write(b"Kokodiv UI file not found.")

        elif parsed_path == "/api/health":
            self._set_headers(200, "application/json")
            resp = {
                "status": "healthy",
                "onnx_loaded": classifier is not None,
                "db_loaded": os.path.exists(DB_PATH)
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif parsed_path == "/api/chat/sessions":
            sessions = retriever.list_chat_sessions()
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({"sessions": sessions}).encode("utf-8"))

        elif parsed_path == "/api/chat/messages":
            session_id = query_params.get("session_id", [""])[0]
            messages = retriever.get_chat_messages(session_id) if session_id else []
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({"messages": messages}).encode("utf-8"))

        elif parsed_path == "/api/crin/manual":
            lang = query_params.get("language", ["en"])[0]
            diseases = ["black_pod", "monilia", "pod_borer", "witches_broom", "cssvd", "frosty_pod", "anthracnose", "mirid", "healthy"]
            manual_items = []
            for d in diseases:
                prof = retriever.get_disease_profile(d, language_code=lang)
                if prof:
                    manual_items.append(prof)
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({"manual": manual_items}, ensure_ascii=False).encode("utf-8"))

        else:
            self._set_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        
        if parsed_path == "/api/chat/sessions":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode('utf-8')) if post_data else {}
            title = payload.get("title", "New Advisory Session")
            lang = payload.get("language", "en")
            session_id = retriever.create_chat_session(title=title, language_code=lang)
            self._set_headers(200, "application/json")
            self.wfile.write(json.dumps({"session_id": session_id}).encode("utf-8"))

        elif parsed_path == "/api/diagnose":
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                payload = json.loads(post_data.decode('utf-8'))

                user_text = payload.get("user_text", "")
                override_lang = payload.get("language", None)
                session_id = payload.get("session_id", None)
                image_bytes = None

                if "image_base64" in payload and payload["image_base64"]:
                    import base64
                    img_str = payload["image_base64"]
                    if "," in img_str:
                        img_str = img_str.split(",")[1]
                    image_bytes = base64.b64decode(img_str)

                # 1. Classification Phase (Photo Mode vs Text-Only Mode)
                has_photo = image_bytes is not None
                if classifier and has_photo:
                    tensor = preprocess_image(image_bytes, target_size=336)
                    cnn_scores = classifier.predict(tensor)
                else:
                    # Text-Only Agronomy Advisory Mode: Search RAG database for matching topics
                    matched_profs = retriever.search_agronomy_topics(user_text, language_code=override_lang or "en")
                    if matched_profs:
                        top_dis = matched_profs[0]["disease_id"]
                        cnn_scores = {top_dis: 0.70, "healthy": 0.15, "black_pod": 0.15}
                    else:
                        cnn_scores = {"healthy": 0.50, "black_pod": 0.25, "mirid": 0.25}

                # 2. Build system prompt & retrieve RAG context
                prompt_data = context_builder.build_full_prompt(
                    cnn_scores=cnn_scores,
                    user_input=user_text,
                    override_language=override_lang
                )

                # 3. Call llama.cpp HTTP server endpoint if available
                llama_response_text = None
                try:
                    chat_url = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8080/v1/chat/completions")
                    req_payload = json.dumps({
                        "messages": [
                            {"role": "system", "content": prompt_data.get("system_prompt", "You are Kokodiv, an offline agronomy doctor.")},
                            {"role": "user", "content": prompt_data["prompt"]}
                        ],
                        "prompt": prompt_data["prompt"],
                        "temperature": 0.2,
                        "max_tokens": 300,
                        "n_predict": 300,
                        "stop": ["USER:", "SYSTEM:"]
                    }).encode('utf-8')

                    req = urllib.request.Request(
                        chat_url,
                        data=req_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=10) as llama_resp:
                        llama_json = json.loads(llama_resp.read().decode('utf-8'))
                        if "choices" in llama_json and len(llama_json["choices"]) > 0:
                            choice = llama_json["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                llama_response_text = choice["message"]["content"].strip()
                            elif "text" in choice:
                                llama_response_text = choice["text"].strip()
                        elif "content" in llama_json:
                            llama_response_text = llama_json.get("content", "").strip()
                except Exception as e:
                    # Fallback to RAG context directly if llama.cpp server is offline
                    top_id = prompt_data["top_candidates"][0]
                    profile = retriever.get_disease_profile(top_id, language_code=prompt_data["detected_language"])
                    if profile:
                        treats = "\n".join([f"• {t}" for t in profile["treatment"]])
                        llama_response_text = (
                            f"**Plant Disease:** {profile['common_name']}\n"
                            f"**What causes it:** {profile['pathogen']}\n\n"
                            f"**How to treat your cocoa trees:**\n{treats}\n\n"
                            f"**How to prevent it:** {profile['prevention']}"
                        )

                # Save to Chat History if session_id provided
                if session_id:
                    retriever.add_chat_message(
                        session_id=session_id,
                        role="user",
                        content=user_text or "[Image Uploaded]",
                        has_image=has_photo
                    )
                    retriever.add_chat_message(
                        session_id=session_id,
                        role="assistant",
                        content=llama_response_text,
                        disease_detected=prompt_data["top_candidates"][0],
                        confidence_score=max(cnn_scores.values())
                    )

                response_payload = {
                    "detected_language": prompt_data["detected_language"],
                    "language_name": prompt_data["language_name"],
                    "confidence_note": prompt_data["confidence_note"],
                    "top_candidates": prompt_data["top_candidates"],
                    "cnn_scores": cnn_scores,
                    "response": llama_response_text,
                    "has_photo": has_photo
                }

                self._set_headers(200, "application/json")
                self.wfile.write(json.dumps(response_payload, ensure_ascii=False).encode("utf-8"))

            except Exception as err:
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({"error": str(err)}).encode("utf-8"))

def run_server(port: int = 8090):
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, KokodivRequestHandler)
    print(f"==================================================")
    print(f" Kokodiv Conversational Application Server Running")
    print(f" URL: http://localhost:{port}")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Kokodiv server.")
        httpd.server_close()

if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    run_server(port_arg)

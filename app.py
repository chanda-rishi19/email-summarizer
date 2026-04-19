from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import spacy

app = Flask(__name__)
CORS(app)

# Load models once at startup
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    text = data.get("text", "").strip()
    mode = data.get("mode", "brief")   # brief | bullets | actions

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # --- Summarization ---
    # BART needs input under ~1024 tokens; truncate long inputs
    max_input = 1024
    words = text.split()
    if len(words) > max_input:
        text = " ".join(words[:max_input])

    result = summarizer(
        text,
        max_length=130,
        min_length=30,
        do_sample=False
    )
    summary = result[0]["summary_text"]

    # --- Entity extraction with spaCy ---
    doc = nlp(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "DATE", "ORG", "GPE") and ent.text not in seen:
            entities.append({"text": ent.text, "label": ent.label_})
            seen.add(ent.text)

    # --- Format output by mode ---
    if mode == "bullets":
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        output = sentences
    elif mode == "actions":
        # Filter sentences with action keywords
        action_words = ["should", "will", "must", "need", "please", "submit", "send", "review", "complete"]
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        output = [s for s in sentences if any(w in s.lower() for w in action_words)] or sentences
    else:
        output = summary   # plain text for "brief"

    return jsonify({
        "summary": output,
        "entities": entities,
        "word_count_in": len(text.split()),
        "word_count_out": len(summary.split())
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
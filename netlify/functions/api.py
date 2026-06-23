import json
import re
import traceback
from pathlib import Path
import tiktoken
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ── CONFIG ────────────────────────────────────────────────────────
NLTK_DIR = Path("/tmp/nltk_data")
MODELS = {
    "gpt-4o": { "label": "GPT-4o", "encoding": "o200k_base", "multiplier": 1.0, "input_cost": 2.50, "output_cost": 10.00, "context_window": 128_000 },
    "claude-sonnet-4": { "label": "Claude Sonnet 4", "encoding": "cl100k_base", "multiplier": 1.05, "input_cost": 3.00, "output_cost": 15.00, "context_window": 200_000 },
    "gemini-2.0-flash": { "label": "Gemini 2.0 Flash", "encoding": "cl100k_base", "multiplier": 0.95, "input_cost": 0.10, "output_cost": 0.40, "context_window": 1_000_000 },
}

# ── LAZY INIT ─────────────────────────────────────────────────────
_INITIALIZED = False
_ENC = {}
_STOP = set()

def _initialize():
    global _INITIALIZED, _ENC, _STOP
    if _INITIALIZED: return
    
    # NLTK Setup
    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(NLTK_DIR))
    
    needed = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    }
    for resource_path, pkg_name in needed.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(pkg_name, download_dir=str(NLTK_DIR), quiet=True)
            
    # Tiktoken Setup
    for mid, cfg in MODELS.items():
        enc_name = cfg["encoding"]
        if enc_name not in _ENC:
            _ENC[enc_name] = tiktoken.get_encoding(enc_name)
            
    _STOP = set(stopwords.words("english"))
    _INITIALIZED = True

# ── DATA ──────────────────────────────────────────────────────────
ACTION_VERBS = {"write", "create", "make", "build", "generate", "explain", "analyze", "describe", "list", "show", "give", "find", "help", "design", "code", "implement", "train", "develop", "compare", "summarize", "detail", "produce", "optimize", "debug", "refactor", "deploy", "test", "translate", "convert", "calculate", "evaluate", "plan", "fix", "configure", "setup", "install", "migrate", "integrate", "solve"}
TECH_TERMS = {"python", "javascript", "typescript", "react", "vue", "angular", "api", "rest", "graphql", "neural", "network", "model", "data", "algorithm", "database", "function", "class", "pytorch", "tensorflow", "sql", "nosql", "html", "css", "json", "yaml", "xml", "llm", "gpt", "bert", "transformer", "vector", "embedding", "docker", "kubernetes", "aws", "azure", "gcp", "linux", "server", "machine", "learning", "deep", "regression", "classification", "clustering", "backend", "frontend", "fullstack", "microservice", "token", "prompt", "inference", "training", "dataset", "feature", "component", "module", "package", "library", "framework"}
QUALITY_MODIFIERS = {"step", "guide", "tutorial", "detailed", "complete", "full", "comprehensive", "advanced", "optimized", "efficient", "structured", "professional", "simple", "basic", "complex", "minimal", "robust", "scalable", "production", "enterprise", "beginner", "expert", "modern", "lightweight", "secure", "performant", "reliable"}
POS_WEIGHTS = {"VB": 0.70, "VBP": 0.70, "VBZ": 0.70, "VBG": 0.65, "VBD": 0.65, "VBN": 0.60, "NN": 0.60, "NNS": 0.60, "NNP": 0.80, "NNPS": 0.80, "JJ": 0.55, "JJR": 0.55, "JJS": 0.55, "RB": 0.35, "RBR": 0.35, "RBS": 0.35, "CD": 0.45}

INTENTS = [
    {"name": "Step-by-step Guide", "icon": "list-ordered", "patterns": [r"step.by.step", r"how to", r"\bguide\b", r"tutorial", r"procedure", r"walk.?through", r"instructions"]},
    {"name": "Code Generation", "icon": "code", "patterns": [r"\bcode\b", r"\bprogram\b", r"\bfunction\b", r"\bscript\b", r"\bimplement\b", r"\bbuild\b", r"\bdevelop\b", r"\bpython\b", r"\bjavascript\b", r"\bapi\b"]},
    {"name": "Creative Writing", "icon": "pen", "patterns": [r"\bwrite\b", r"\bstory\b", r"\bessay\b", r"\bblog\b", r"\barticle\b", r"\bpoem\b", r"\bnarrative\b", r"\bcreative\b"]},
    {"name": "Explanation / ELI5", "icon": "info", "patterns": [r"\bexplain\b", r"what is", r"\bdescribe\b", r"\bdefine\b", r"\bmeaning\b", r"\bunderstand\b", r"tell me about"]},
    {"name": "List / Enumeration", "icon": "list", "patterns": [r"\blist\b", r"\benumerate\b", r"examples of", r"top \d", r"best \d", r"give me.*example"]},
    {"name": "Analysis / Research", "icon": "chart", "patterns": [r"\banalyze\b", r"\banalyse\b", r"\bcompare\b", r"\bevaluate\b", r"\bassess\b", r"pros and cons", r"\bdifference\b", r"\bversus\b"]},
    {"name": "Q&A / Factual", "icon": "help", "patterns": [r"\bwho\b", r"\bwhat\b", r"\bwhen\b", r"\bwhere\b", r"\bwhy\b", r"how many", r"how much"]},
]

FLOWCHARTS = {
    "Step-by-step Guide": [{"label": "USER PROMPT", "type": "start"}, {"label": "Intent Parse", "type": "intent"}, {"label": "Instruction Decode", "type": "step"}, {"label": "Context Window Scan", "type": "step"}, {"label": "RAG Retrieval", "type": "rag"}, {"label": "Enumerate Steps", "type": "step"}, {"label": "Draft Each Step Body", "type": "step"}, {"label": "Conclusion Block", "type": "output"}],
    "Code Generation": [{"label": "USER PROMPT", "type": "start"}, {"label": "Intent Parse", "type": "intent"}, {"label": "Language Detection", "type": "step"}, {"label": "Pattern Recall", "type": "step"}, {"label": "RAG: Codebase Docs", "type": "rag"}, {"label": "Core Logic Draft", "type": "step"}, {"label": "Error Handling", "type": "step"}, {"label": "Comments + Docs", "type": "step"}, {"label": "Usage Example", "type": "output"}],
    "Creative Writing": [{"label": "USER PROMPT", "type": "start"}, {"label": "Intent Parse", "type": "intent"}, {"label": "Tone + Style Detect", "type": "step"}, {"label": "Setting Build", "type": "step"}, {"label": "Narrative Arc Plan", "type": "step"}, {"label": "Draft Content", "type": "step"}, {"label": "Polish + Voice Refine", "type": "output"}],
    "Explanation / ELI5": [{"label": "USER PROMPT", "type": "start"}, {"label": "Intent Parse", "type": "intent"}, {"label": "Concept Identification", "type": "step"}, {"label": "RAG: Context Fetch", "type": "rag"}, {"label": "Sub-concept Breakdown", "type": "step"}, {"label": "Analogy Generation", "type": "step"}, {"label": "Summary Block", "type": "output"}],
    "Analysis / Research": [{"label": "USER PROMPT", "type": "start"}, {"label": "Intent Parse", "type": "intent"}, {"label": "Scope Definition", "type": "step"}, {"label": "RAG: Source Fetch", "type": "rag"}, {"label": "Dimension Compare", "type": "step"}, {"label": "Structure Findings", "type": "step"}, {"label": "Verdict Block", "type": "output"}],
    "List / Enumeration": [{"label": "USER PROMPT", "type": "start"}, {"label": "Intent Parse", "type": "intent"}, {"label": "Category Detection", "type": "step"}, {"label": "RAG: Item Retrieval", "type": "rag"}, {"label": "Sort by Relevance", "type": "step"}, {"label": "Format Each Item", "type": "step"}, {"label": "Final List Output", "type": "output"}],
    "Q&A / Factual": [{"label": "USER PROMPT", "type": "start"}, {"label": "Intent Parse", "type": "intent"}, {"label": "Question Parsing", "type": "step"}, {"label": "RAG: Fact Fetch", "type": "rag"}, {"label": "Answer Construction", "type": "step"}, {"label": "Direct Response", "type": "output"}],
}

# ── API LOGIC ────────────────────────────────────────────────────
def handle_tokens(prompt):
    text = prompt.strip()
    if not text: return {}
    words = text.split()
    word_count, char_count = len(words), len(text)
    results = {}
    for mid, cfg in MODELS.items():
        enc = _ENC[cfg["encoding"]]
        raw = len(enc.encode(text))
        pt = max(1, round(raw * cfg["multiplier"]))
        o_min = round(pt * 1.8); o_max = round(pt * 7.5); o_mid = (o_min + o_max) // 2
        results[mid] = { "label": cfg["label"], "promptTokens": pt, "wordCount": word_count, "charCount": char_count, "outputMin": o_min, "outputMax": o_max, "outputMid": o_mid, "total": pt + o_max, "cost": round((pt / 1e6) * cfg["input_cost"] + (o_mid / 1e6) * cfg["output_cost"], 6), "contextWindow": cfg["context_window"], "inputCostPer1M": cfg["input_cost"], "outputCostPer1M": cfg["output_cost"] }
    return results

def handle_weights(prompt):
    text = prompt.strip()
    if not text: return {"words": []}
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    freq = {}
    for tok in tokens:
        c = re.sub(r"[^a-z]", "", tok.lower())
        if c and c not in _STOP: freq[c] = freq.get(c, 0) + 1
    max_freq = max(freq.values()) if freq else 1
    words = []
    for word, pos in tagged:
        clean = re.sub(r"[^a-zA-Z0-9]", "", word).lower()
        if not clean:
            words.append({"word": word, "score": 0.05})
            continue
        base = POS_WEIGHTS.get(pos, 0.15)
        if clean in _STOP: base = 0.08
        else:
            if clean in ACTION_VERBS: base = max(base, 0.92)
            elif clean in TECH_TERMS: base = max(base, 0.85)
            elif clean in QUALITY_MODIFIERS: base = max(base, 0.72)
            f = freq.get(clean, 0) / max_freq
            base = min(1.0, base + f * 0.08)
        words.append({"word": word, "score": round(base, 3)})
    return {"words": words}

def handle_intent(prompt):
    text = prompt.strip().lower()
    results = []
    for i, intent in enumerate(INTENTS):
        matches = sum(1 for p in intent["patterns"] if re.search(p, text))
        n = len(intent["patterns"])
        score = min(0.98, 0.45 + (matches / n) * 0.50) if matches > 0 else max(0.02, 0.12 - i * 0.014)
        results.append({ "name": intent["name"], "score": round(score, 2), "icon": intent["icon"] })
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"intents": results}

def handle_flowchart(prompt):
    intent_data = handle_intent(prompt)
    top = intent_data["intents"][0]["name"]
    nodes = FLOWCHARTS.get(top, FLOWCHARTS["Q&A / Factual"])
    return {"intent": top, "nodes": nodes}

# ── LAMBDA HANDLER ───────────────────────────────────────────────
CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}

def handler(event, context):
    try:
        # Handle CORS preflight
        if event.get("httpMethod") == "OPTIONS":
            return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

        if event.get("httpMethod") != "POST":
            return {"statusCode": 405, "headers": CORS_HEADERS, "body": json.dumps({"error": "Method not allowed"})}

        _initialize()

        # Extract route from path — works whether the path is:
        #   /api/tokens  (via rewrite)  OR  /.netlify/functions/api/tokens  (direct call)
        path = event.get("path", "")
        # Strip trailing slash and take the last non-empty segment
        segments = [s for s in path.split("/") if s]
        sub_path = segments[-1] if segments else ""

        # Also check queryStringParameters as a fallback (?action=tokens)
        if sub_path in ("api",):
            qs = event.get("queryStringParameters") or {}
            sub_path = qs.get("action", sub_path)

        body_raw = event.get("body") or "{}"
        body = json.loads(body_raw)
        prompt = body.get("prompt", "")

        if sub_path == "tokens":
            result = handle_tokens(prompt)
        elif sub_path == "weights":
            result = handle_weights(prompt)
        elif sub_path == "intent":
            result = handle_intent(prompt)
        elif sub_path == "flowchart":
            result = handle_flowchart(prompt)
        else:
            return {"statusCode": 404, "headers": CORS_HEADERS, "body": json.dumps({"error": f"Unknown endpoint: {sub_path}"})}

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps(result),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({
                "error": str(e),
                "trace": traceback.format_exc(),
            }),
        }

import { get_encoding } from '@dqbd/tiktoken';
import nlp from 'compromise';

const MODELS = {
  'gpt-4o': {
    label: 'GPT-4o',
    encoding: 'o200k_base',
    multiplier: 1.0,
    input_cost: 2.5,
    output_cost: 10.0,
    context_window: 128_000,
  },
  'claude-sonnet-4': {
    label: 'Claude Sonnet 4',
    encoding: 'cl100k_base',
    multiplier: 1.05,
    input_cost: 3.0,
    output_cost: 15.0,
    context_window: 200_000,
  },
  'gemini-2.0-flash': {
    label: 'Gemini 2.0 Flash',
    encoding: 'cl100k_base',
    multiplier: 0.95,
    input_cost: 0.1,
    output_cost: 0.4,
    context_window: 1_000_000,
  },
};

const ACTION_VERBS = new Set([
  'write', 'create', 'make', 'build', 'generate', 'explain', 'analyze', 'describe', 'list', 'show',
  'give', 'find', 'help', 'design', 'code', 'implement', 'train', 'develop', 'compare', 'summarize',
  'detail', 'produce', 'optimize', 'debug', 'refactor', 'deploy', 'test', 'translate', 'convert',
  'calculate', 'evaluate', 'plan', 'fix', 'configure', 'setup', 'install', 'migrate', 'integrate', 'solve',
]);

const TECH_TERMS = new Set([
  'python', 'javascript', 'typescript', 'react', 'vue', 'angular', 'api', 'rest', 'graphql', 'neural',
  'network', 'model', 'data', 'algorithm', 'database', 'function', 'class', 'pytorch', 'tensorflow',
  'sql', 'nosql', 'html', 'css', 'json', 'yaml', 'xml', 'llm', 'gpt', 'bert', 'transformer', 'vector',
  'embedding', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'linux', 'server', 'machine', 'learning',
  'deep', 'regression', 'classification', 'clustering', 'backend', 'frontend', 'fullstack', 'microservice',
  'token', 'prompt', 'inference', 'training', 'dataset', 'feature', 'component', 'module', 'package',
  'library', 'framework',
]);

const QUALITY_MODIFIERS = new Set([
  'step', 'guide', 'tutorial', 'detailed', 'complete', 'full', 'comprehensive', 'advanced', 'optimized',
  'efficient', 'structured', 'professional', 'simple', 'basic', 'complex', 'minimal', 'robust', 'scalable',
  'production', 'enterprise', 'beginner', 'expert', 'modern', 'lightweight', 'secure', 'performant', 'reliable',
]);

const STOP = new Set([
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
  'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
  'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
  'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
  'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
  'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
  'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
  'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
  'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
]);

const TAG_WEIGHTS = [
  ['Verb', 0.7],
  ['Noun', 0.6],
  ['ProperNoun', 0.8],
  ['Adjective', 0.55],
  ['Adverb', 0.35],
  ['Value', 0.45],
];

const INTENTS = [
  { name: 'Step-by-step Guide', icon: 'list-ordered', patterns: [/step.by.step/i, /how to/i, /\bguide\b/i, /tutorial/i, /procedure/i, /walk.?through/i, /instructions/i] },
  { name: 'Code Generation', icon: 'code', patterns: [/\bcode\b/i, /\bprogram\b/i, /\bfunction\b/i, /\bscript\b/i, /\bimplement\b/i, /\bbuild\b/i, /\bdevelop\b/i, /\bpython\b/i, /\bjavascript\b/i, /\bapi\b/i] },
  { name: 'Creative Writing', icon: 'pen', patterns: [/\bwrite\b/i, /\bstory\b/i, /\bessay\b/i, /\bblog\b/i, /\barticle\b/i, /\bpoem\b/i, /\bnarrative\b/i, /\bcreative\b/i] },
  { name: 'Explanation / ELI5', icon: 'info', patterns: [/\bexplain\b/i, /what is/i, /\bdescribe\b/i, /\bdefine\b/i, /\bmeaning\b/i, /\bunderstand\b/i, /tell me about/i] },
  { name: 'List / Enumeration', icon: 'list', patterns: [/\blist\b/i, /\benumerate\b/i, /examples of/i, /top \d/i, /best \d/i, /give me.*example/i] },
  { name: 'Analysis / Research', icon: 'chart', patterns: [/\banalyze\b/i, /\banalyse\b/i, /\bcompare\b/i, /\bevaluate\b/i, /\bassess\b/i, /pros and cons/i, /\bdifference\b/i, /\bversus\b/i] },
  { name: 'Q&A / Factual', icon: 'help', patterns: [/\bwho\b/i, /\bwhat\b/i, /\bwhen\b/i, /\bwhere\b/i, /\bwhy\b/i, /how many/i, /how much/i] },
];

const FLOWCHARTS = {
  'Step-by-step Guide': [
    { label: 'USER PROMPT', type: 'start' }, { label: 'Intent Parse', type: 'intent' }, { label: 'Instruction Decode', type: 'step' },
    { label: 'Context Window Scan', type: 'step' }, { label: 'RAG Retrieval', type: 'rag' }, { label: 'Enumerate Steps', type: 'step' },
    { label: 'Draft Each Step Body', type: 'step' }, { label: 'Conclusion Block', type: 'output' },
  ],
  'Code Generation': [
    { label: 'USER PROMPT', type: 'start' }, { label: 'Intent Parse', type: 'intent' }, { label: 'Language Detection', type: 'step' },
    { label: 'Pattern Recall', type: 'step' }, { label: 'RAG: Codebase Docs', type: 'rag' }, { label: 'Core Logic Draft', type: 'step' },
    { label: 'Error Handling', type: 'step' }, { label: 'Comments + Docs', type: 'step' }, { label: 'Usage Example', type: 'output' },
  ],
  'Creative Writing': [
    { label: 'USER PROMPT', type: 'start' }, { label: 'Intent Parse', type: 'intent' }, { label: 'Tone + Style Detect', type: 'step' },
    { label: 'Setting Build', type: 'step' }, { label: 'Narrative Arc Plan', type: 'step' }, { label: 'Draft Content', type: 'step' },
    { label: 'Polish + Voice Refine', type: 'output' },
  ],
  'Explanation / ELI5': [
    { label: 'USER PROMPT', type: 'start' }, { label: 'Intent Parse', type: 'intent' }, { label: 'Concept Identification', type: 'step' },
    { label: 'RAG: Context Fetch', type: 'rag' }, { label: 'Sub-concept Breakdown', type: 'step' }, { label: 'Analogy Generation', type: 'step' },
    { label: 'Summary Block', type: 'output' },
  ],
  'Analysis / Research': [
    { label: 'USER PROMPT', type: 'start' }, { label: 'Intent Parse', type: 'intent' }, { label: 'Scope Definition', type: 'step' },
    { label: 'RAG: Source Fetch', type: 'rag' }, { label: 'Dimension Compare', type: 'step' }, { label: 'Structure Findings', type: 'step' },
    { label: 'Verdict Block', type: 'output' },
  ],
  'List / Enumeration': [
    { label: 'USER PROMPT', type: 'start' }, { label: 'Intent Parse', type: 'intent' }, { label: 'Category Detection', type: 'step' },
    { label: 'RAG: Item Retrieval', type: 'rag' }, { label: 'Sort by Relevance', type: 'step' }, { label: 'Format Each Item', type: 'step' },
    { label: 'Final List Output', type: 'output' },
  ],
  'Q&A / Factual': [
    { label: 'USER PROMPT', type: 'start' }, { label: 'Intent Parse', type: 'intent' }, { label: 'Question Parsing', type: 'step' },
    { label: 'RAG: Fact Fetch', type: 'rag' }, { label: 'Answer Construction', type: 'step' }, { label: 'Direct Response', type: 'output' },
  ],
};

const encCache = {};

function getEnc(name) {
  if (!encCache[name]) encCache[name] = get_encoding(name);
  return encCache[name];
}

function tokenize(text) {
  return text.match(/\w+(?:['’]\w+)*|[^\w\s]/g) || [];
}

function posWeight(tags = []) {
  for (const [tag, weight] of TAG_WEIGHTS) {
    if (tags.includes(tag)) return weight;
  }
  return 0.15;
}

function buildTagMap(text) {
  const map = new Map();
  for (const chunk of nlp(text).terms().json()) {
    for (const term of chunk.terms) {
      const key = (term.normal || term.text).toLowerCase();
      if (!map.has(key)) map.set(key, term.tags || []);
    }
  }
  return map;
}

export function handleTokens(prompt) {
  const text = prompt.trim();
  if (!text) return {};

  const words = text.split(/\s+/);
  const wordCount = words.length;
  const charCount = text.length;
  const results = {};

  for (const [mid, cfg] of Object.entries(MODELS)) {
    const enc = getEnc(cfg.encoding);
    const raw = enc.encode(text).length;
    const pt = Math.max(1, Math.round(raw * cfg.multiplier));
    const oMin = Math.round(pt * 1.8);
    const oMax = Math.round(pt * 7.5);
    const oMid = Math.floor((oMin + oMax) / 2);

    results[mid] = {
      label: cfg.label,
      promptTokens: pt,
      wordCount,
      charCount,
      outputMin: oMin,
      outputMax: oMax,
      outputMid: oMid,
      total: pt + oMax,
      cost: Number(((pt / 1e6) * cfg.input_cost + (oMid / 1e6) * cfg.output_cost).toFixed(6)),
      contextWindow: cfg.context_window,
      inputCostPer1M: cfg.input_cost,
      outputCostPer1M: cfg.output_cost,
    };
  }

  return results;
}

export function handleWeights(prompt) {
  const text = prompt.trim();
  if (!text) return { words: [] };

  const tokens = tokenize(text);
  const tagMap = buildTagMap(text);

  const freq = {};
  for (const tok of tokens) {
    const c = tok.toLowerCase().replace(/[^a-z]/g, '');
    if (c && !STOP.has(c)) freq[c] = (freq[c] || 0) + 1;
  }
  const maxFreq = Math.max(...Object.values(freq), 1);

  const words = tokens.map((word) => {
    const clean = word.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    if (!clean) return { word, score: 0.05 };

    let base = posWeight(tagMap.get(clean));
    if (STOP.has(clean)) {
      base = 0.08;
    } else {
      if (ACTION_VERBS.has(clean)) base = Math.max(base, 0.92);
      else if (TECH_TERMS.has(clean)) base = Math.max(base, 0.85);
      else if (QUALITY_MODIFIERS.has(clean)) base = Math.max(base, 0.72);
      const f = (freq[clean] || 0) / maxFreq;
      base = Math.min(1, base + f * 0.08);
    }

    return { word, score: Number(base.toFixed(3)) };
  });

  return { words };
}

export function handleIntent(prompt) {
  const text = prompt.trim().toLowerCase();
  const results = INTENTS.map((intent, i) => {
    const matches = intent.patterns.reduce((n, pattern) => n + (pattern.test(text) ? 1 : 0), 0);
    const n = intent.patterns.length;
    const score = matches > 0
      ? Math.min(0.98, 0.45 + (matches / n) * 0.5)
      : Math.max(0.02, 0.12 - i * 0.014);
    return { name: intent.name, score: Number(score.toFixed(2)), icon: intent.icon };
  });

  results.sort((a, b) => b.score - a.score);
  return { intents: results };
}

export function handleFlowchart(prompt) {
  const intentData = handleIntent(prompt);
  const top = intentData.intents[0].name;
  const nodes = FLOWCHARTS[top] || FLOWCHARTS['Q&A / Factual'];
  return { intent: top, nodes };
}

import {
  handleTokens,
  handleWeights,
  handleIntent,
  handleFlowchart,
} from './lib/engine.mjs';

const CORS_HEADERS = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
};

function resolveEndpoint(req) {
  const url = new URL(req.url);
  const endpoint = url.searchParams.get('endpoint');
  if (endpoint) return endpoint;

  const originalPath = req.headers.get('x-netlify-original-path');
  if (originalPath) {
    const segments = originalPath.split('/').filter(Boolean);
    if (segments.length && segments[segments.length - 1] !== 'api') {
      return segments[segments.length - 1];
    }
  }

  const segments = url.pathname.split('/').filter(Boolean);
  if (segments.length && segments[segments.length - 1] !== 'api') {
    return segments[segments.length - 1];
  }

  return '';
}

function jsonResponse(status, data) {
  return new Response(JSON.stringify(data), {
    status,
    headers: CORS_HEADERS,
  });
}

export default async (req) => {
  try {
    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    if (req.method !== 'POST') {
      return jsonResponse(405, { error: 'Method not allowed' });
    }

    const subPath = resolveEndpoint(req);
    const body = await req.json().catch(() => ({}));
    const prompt = body.prompt || '';

    let result;
    switch (subPath) {
      case 'tokens':
        result = handleTokens(prompt);
        break;
      case 'weights':
        result = handleWeights(prompt);
        break;
      case 'intent':
        result = handleIntent(prompt);
        break;
      case 'flowchart':
        result = handleFlowchart(prompt);
        break;
      default:
        return jsonResponse(404, { error: `Unknown endpoint: ${subPath}` });
    }

    return jsonResponse(200, result);
  } catch (error) {
    return jsonResponse(500, {
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

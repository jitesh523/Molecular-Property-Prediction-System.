# Browser / TypeScript SDK

A small zero-dependency wrapper around `fetch` for calling the **molprop** API
from a browser, Node, Deno, or Bun. The whole client is one file (≈80 lines)
and ships as both a JS and a TS snippet — no npm install required, just paste.

---

## TypeScript

```ts
// molprop-client.ts
export class MolpropAPIError extends Error {
  constructor(public status: number, public detail: unknown, message: string) {
    super(message);
  }
}

export interface MolpropClientOptions {
  baseUrl?: string;     // default: same-origin
  timeoutMs?: number;   // default: 30_000
  apiKey?: string;      // sent as X-API-Key if set
}

export class MolpropClient {
  private baseUrl: string;
  private timeoutMs: number;
  private apiKey?: string;

  constructor(opts: MolpropClientOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? "").replace(/\/$/, "");
    this.timeoutMs = opts.timeoutMs ?? 30_000;
    this.apiKey = opts.apiKey;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const ctrl = new AbortController();
    const id = setTimeout(() => ctrl.abort(), this.timeoutMs);
    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal: ctrl.signal,
        headers: {
          "Content-Type": "application/json",
          ...(this.apiKey ? { "X-API-Key": this.apiKey } : {}),
          ...(init.headers ?? {}),
        },
      });
      const text = await resp.text();
      const data = text ? JSON.parse(text) : null;
      if (!resp.ok) {
        throw new MolpropAPIError(resp.status, data, data?.detail ?? resp.statusText);
      }
      return data as T;
    } finally {
      clearTimeout(id);
    }
  }

  // ── Cheminformatics ──────────────────────────────────────────────────
  health()                   { return this.request<{ status: string }>("/health"); }
  version()                  { return this.request<{ version: string }>("/version"); }
  scaffold(smiles: string)   { return this.request<any>("/scaffold",   { method: "POST", body: JSON.stringify({ smiles }) }); }
  alerts(smiles: string)     { return this.request<any>("/alerts",     { method: "POST", body: JSON.stringify({ smiles }) }); }
  isomers(smiles: string)    { return this.request<any>("/isomers",    { method: "POST", body: JSON.stringify({ smiles }) }); }
  mcs(a: string, b: string)  { return this.request<any>("/mcs",        { method: "POST", body: JSON.stringify({ smiles_a: a, smiles_b: b }) }); }
  rgroups(core: string, smilesList: string[]) {
    return this.request<any>("/rgroups", { method: "POST", body: JSON.stringify({ core, smiles_list: smilesList }) });
  }
  react(substrates: string[][], opts: { smarts?: string; named?: string } = {}) {
    return this.request<any>("/react", { method: "POST", body: JSON.stringify({ substrates, ...opts }) });
  }
  mmp(smilesList: string[], opts: { max_substituent_atoms?: number; max_pairs?: number } = {}) {
    return this.request<any>("/mmp", { method: "POST", body: JSON.stringify({ smiles_list: smilesList, ...opts }) });
  }
  depict(smiles: string, opts: { width?: number; height?: number; highlight_smarts?: string } = {}) {
    return this.request<any>("/depict", { method: "POST", body: JSON.stringify({ smiles, ...opts }) });
  }
  predict(smiles: string)   { return this.request<any>("/predict", { method: "POST", body: JSON.stringify({ smiles }) }); }
}
```

---

## Plain JavaScript (no build step)

```js
// molprop-client.js
class MolpropClient {
  constructor({ baseUrl = "", timeoutMs = 30_000, apiKey } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.timeoutMs = timeoutMs;
    this.apiKey = apiKey;
  }
  async _req(path, init = {}) {
    const ctrl = new AbortController();
    const id = setTimeout(() => ctrl.abort(), this.timeoutMs);
    try {
      const resp = await fetch(this.baseUrl + path, {
        ...init,
        signal: ctrl.signal,
        headers: {
          "Content-Type": "application/json",
          ...(this.apiKey && { "X-API-Key": this.apiKey }),
          ...(init.headers || {}),
        },
      });
      const data = await resp.json().catch(() => null);
      if (!resp.ok) throw Object.assign(new Error(data?.detail || resp.statusText), { status: resp.status, data });
      return data;
    } finally { clearTimeout(id); }
  }
  health()                    { return this._req("/health"); }
  scaffold(smiles)            { return this._req("/scaffold",  { method: "POST", body: JSON.stringify({ smiles }) }); }
  depict(smiles, opts = {})   { return this._req("/depict",    { method: "POST", body: JSON.stringify({ smiles, ...opts }) }); }
  mmp(smilesList, opts = {})  { return this._req("/mmp",       { method: "POST", body: JSON.stringify({ smiles_list: smilesList, ...opts }) }); }
  react(substrates, opts = {}){ return this._req("/react",     { method: "POST", body: JSON.stringify({ substrates, ...opts }) }); }
}
window.MolpropClient = MolpropClient; // for <script> tags
```

---

## Usage examples

### Browser — drop-in `<script>`

```html
<script src="molprop-client.js"></script>
<script>
  const mp = new MolpropClient({ baseUrl: "https://api.example.com" });
  mp.depict("CC(=O)Oc1ccccc1C(=O)O", { width: 500, highlight_smarts: "C(=O)O" })
    .then(r => document.getElementById("mol").innerHTML = r.svg);
</script>
```

### Node 18+ / Deno / Bun

```ts
import { MolpropClient } from "./molprop-client.ts";

const mp = new MolpropClient({ baseUrl: "http://localhost:8000" });
const v = await mp.version();
console.log("API version:", v.version);

const result = await mp.mmp(
  ["c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(C)cc1", "c1ccc(F)cc1"],
  { max_pairs: 50 }
);
console.log(`Found ${result.n_pairs} matched-molecular pairs`);
```

### React component — live 2D structure as the user types

```tsx
import { useEffect, useState } from "react";
import { MolpropClient } from "./molprop-client";

const mp = new MolpropClient();

export function MoleculePreview({ smiles }: { smiles: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  useEffect(() => {
    if (!smiles) return setSvg(null);
    const ctl = new AbortController();
    mp.depict(smiles, { width: 320, height: 240 })
      .then(r => !ctl.signal.aborted && setSvg(r.svg))
      .catch(() => setSvg(null));
    return () => ctl.abort();
  }, [smiles]);
  return svg ? <div dangerouslySetInnerHTML={{ __html: svg }} /> : null;
}
```

---

## Error handling

All non-2xx responses raise `MolpropAPIError` (TS) or a regular `Error` with
`.status` + `.data` (JS). The two most useful status codes:

| Status | Meaning                                    | Recommendation                              |
| -----: | ------------------------------------------ | ------------------------------------------- |
|  `422` | Invalid SMILES or SMARTS, bad request body | Show error inline next to the input field   |
|  `429` | Rate-limit exceeded (token bucket)         | Honour the `Retry-After` response header    |

## Rate limit headers

When the rate-limit middleware is enabled (default 120 req / 60 s per IP),
every successful response carries:

```
X-RateLimit-Limit:     120
X-RateLimit-Remaining: 117
```

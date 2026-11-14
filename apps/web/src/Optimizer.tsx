import React, { useEffect, useState } from 'react';

type Change = { kind: 'applied' | 'suggestion' | 'ai'; title: string; why: string; impact: 'high' | 'medium' | 'low'; before: string; after: string };
type Result = {
  ok: boolean; dialect: string; original_sql: string; optimized_sql: string; changed: boolean;
  changes: Change[]; error: string | null; ai_sql: string | null;
};

const EXAMPLE = `SELECT DISTINCT o.status, COUNT(*)
FROM orders o, customers c
WHERE o.customer_id = c.id
  AND 1 = 1
  AND (o.status = 'paid' OR o.status = 'shipped')
  AND YEAR(o.created_at) = 2024
GROUP BY o.status
HAVING o.status <> 'cancelled' AND COUNT(*) > 5`;

const GROUPS: { kind: Change['kind']; title: string; note: string }[] = [
  { kind: 'applied', title: 'What we changed', note: 'Guaranteed: these rewrites return the same rows on any data.' },
  { kind: 'suggestion', title: 'Worth checking', note: 'Potential: not applied, because the benefit depends on your data, indexes or intent.' },
  { kind: 'ai', title: 'AI ideas', note: 'From a language model. Not verified, test before using.' },
];

const CopyButton: React.FC<{ text: string }> = ({ text }) => {
  const [done, setDone] = useState(false);
  return (
    <button className="ghost" onClick={() => { navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>
      {done ? 'Copied' : 'Copy'}
    </button>
  );
};

export const Optimizer: React.FC = () => {
  const [sql, setSql] = useState(EXAMPLE);
  const [dialect, setDialect] = useState('postgres');
  const [dialects, setDialects] = useState<Record<string, string>>({ postgres: 'PostgreSQL' });
  const [aiOn, setAiOn] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [res, setRes] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);
  const [netError, setNetError] = useState<string | null>(null);
  const [schema, setSchema] = useState('');
  const [showSchema, setShowSchema] = useState(false);

  useEffect(() => {
    fetch('/api/optimize/meta').then((r) => r.json())
      .then((m) => { setDialects(m.dialects); setAiOn(m.ai); setOnline(true); })
      .catch(() => setOnline(false));
  }, []);

  const run = async () => {
    setBusy(true); setNetError(null);
    try {
      const r = await fetch('/api/optimize', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sql, dialect, use_ai: aiOn, schema_ddl: schema }),
      });
      if (!r.ok) throw new Error(r.statusText);
      setRes(await r.json()); setShowOriginal(false);
    } catch {
      setNetError('Could not reach the optimizer. Is the backend running on port 8000?');
    } finally { setBusy(false); }
  };

  const applied = res?.changes.filter((c) => c.kind === 'applied').length ?? 0;

  return (
    <div className="page">
      <header className="head">
        <span className="mark"><i />QueryOpt</span>
        <span className="meta">
          {online === false ? 'backend offline' : aiOn ? 'rules + AI' : 'rules only'}
        </span>
      </header>

      <section className="hero">
        <h1>Make your SQL faster.</h1>
        <p>Paste a query. Get a cleaner, faster version and a plain explanation of every change.</p>
      </section>

      <div className="box">
        <textarea
          className="editor" spellCheck={false} value={sql} placeholder="Paste your SQL here"
          onChange={(e) => setSql(e.target.value)}
          onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run(); }}
        />
        <div className="bar">
          <select value={dialect} onChange={(e) => setDialect(e.target.value)} aria-label="SQL dialect">
            {Object.entries(dialects).map(([id, name]) => <option key={id} value={id}>{name}</option>)}
          </select>
          <button className={`schema-toggle${showSchema || schema ? ' on' : ''}`} onClick={() => setShowSchema(!showSchema)}>
            {schema.trim() ? 'Schema added' : 'Add schema'}
          </button>
          <span className="hint">Ctrl + Enter</span>
          <button className="go" onClick={run} disabled={busy || !sql.trim()}>{busy ? 'Optimizing…' : 'Optimize'}</button>
        </div>
      </div>

      {showSchema && (
        <div className="box schema-box">
          <div className="box-head"><span className="label">Schema (optional)</span><button className="ghost" onClick={() => setShowSchema(false)}>Done</button></div>
          <p className="schema-help">Paste your CREATE TABLE and CREATE INDEX statements. Existing indexes and primary keys are then skipped in the advice, and SELECT * can be expanded. Nothing connects to your database.</p>
          <textarea className="editor" style={{ minHeight: 140 }} spellCheck={false} value={schema}
            placeholder={'CREATE TABLE orders (\n  order_id INT PRIMARY KEY,\n  customer_id INT,\n  created_at DATETIME,\n  KEY (customer_id)\n);'}
            onChange={(e) => setSchema(e.target.value)} />
        </div>
      )}

      {netError && <div className="note err">{netError}</div>}
      {res && !res.ok && <div className="note err">{res.error}</div>}

      {res?.ok && (
        <>
          <div className="verdict">
            {applied > 0
              ? <><b>{applied}</b> improvement{applied > 1 ? 's' : ''} applied.</>
              : <>Already in good shape. Nothing to rewrite safely.</>}
          </div>

          <div className="box">
            <div className="box-head">
              <div className="switch">
                <button className={!showOriginal ? 'on' : ''} onClick={() => setShowOriginal(false)}>Optimized</button>
                <button className={showOriginal ? 'on' : ''} onClick={() => setShowOriginal(true)}>Original</button>
              </div>
              <CopyButton text={showOriginal ? res.original_sql : res.optimized_sql} />
            </div>
            <pre className="code">{showOriginal ? res.original_sql : res.optimized_sql}</pre>
          </div>

          {GROUPS.map((g) => {
            const items = res.changes.filter((c) => c.kind === g.kind);
            if (!items.length) return null;
            return (
              <section className="group" key={g.kind}>
                <h2>{g.title}</h2>
                <p className="group-note">{g.note}</p>
                <ol>
                  {items.map((c, i) => (
                    <li key={i} className="item">
                      <span className={`impact ${c.impact}`} title={`${c.impact} impact`}>{c.impact}</span>
                      <div>
                        <h3>{c.title}</h3>
                        <p>{c.why}</p>
                        {c.before && (
                          <div className="diff">
                            <code className={c.after ? 'old' : 'plain'}>{c.before}</code>
                            {c.after && <code className="new">{c.after}</code>}
                          </div>
                        )}
                      </div>
                    </li>
                  ))}
                </ol>
                {g.kind === 'ai' && res.ai_sql && (
                  <div className="box" style={{ marginTop: 14 }}>
                    <div className="box-head"><span className="label">AI version</span><CopyButton text={res.ai_sql} /></div>
                    <pre className="code">{res.ai_sql}</pre>
                  </div>
                )}
              </section>
            );
          })}
        </>
      )}

      <footer className="foot">Nothing is executed. Impact levels are estimates of what each change usually enables.</footer>
    </div>
  );
};

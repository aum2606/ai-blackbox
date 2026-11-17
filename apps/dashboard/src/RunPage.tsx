import { useEffect, useState, type ReactNode } from "react";
import { ApiRequestError, fetchRun } from "./api";
import type { InvocationOut, RetrievalOut, RunOut, SpanOut } from "./types";

type State =
  | { kind: "loading" }
  | { kind: "ready"; run: RunOut }
  | { kind: "failed"; code: string; message: string };

const NOT_CAPTURED = "not captured";

function Observed() {
  return <span className="badge">observed</span>;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h2>
        {title} <Observed />
      </h2>
      {children}
    </section>
  );
}

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}

function Correctness({ isCorrect }: { isCorrect: boolean | null }) {
  if (isCorrect === null) return <strong>– Not determined</strong>;
  return isCorrect ? <strong>✔ Correct</strong> : <strong>✖ Incorrect</strong>;
}

function Answer({ run }: { run: RunOut }) {
  const { answer } = run;
  return (
    <Section title="Answer">
      <dl>
        <Field
          label="Answer"
          value={answer.captured && answer.text !== null ? answer.text : "content not captured (opt-in was not enabled)"}
        />
        <Field label="Answer hash (SHA-256)" value={<code>{answer.sha256 ?? NOT_CAPTURED}</code>} />
        <Field label="Expected (ground truth)" value={run.expected_answer} />
        <Field label="Result" value={<Correctness isCorrect={run.is_correct} />} />
        <Field label="Run status" value={run.error_type ? `${run.status} (${run.error_type})` : run.status} />
      </dl>
    </Section>
  );
}

function RetrievedDocument({ retrievals }: { retrievals: RetrievalOut[] }) {
  return (
    <Section title="Retrieved document">
      {retrievals.length === 0 && <p>No document recorded for this run.</p>}
      {retrievals.map(({ rank, score, document }) => (
        <dl key={document.id}>
          <Field label="Rank / score" value={`${rank} / ${score}`} />
          <Field label="Document" value={`${document.doc_key}, version ${document.version}`} />
          <Field
            label="Validity"
            value={
              document.valid_to === null
                ? `✔ Current version (valid from ${document.valid_from})`
                : `⚠ Superseded version (valid ${document.valid_from} to ${document.valid_to})`
            }
          />
          <Field label="Document hash (SHA-256)" value={<code>{document.sha256}</code>} />
        </dl>
      ))}
    </Section>
  );
}

function Invocations({ invocations }: { invocations: InvocationOut[] }) {
  return (
    <Section title="Model invocation">
      {invocations.length === 0 && <p>No model invocation recorded for this run.</p>}
      {invocations.map((inv) => (
        <dl key={inv.id}>
          <Field label="Invocation ID" value={<code>{inv.id}</code>} />
          <Field label="Model" value={`${inv.model_id} (tokenizer: ${inv.tokenizer_revision})`} />
          <Field label="Model revision" value={<code>{inv.model_revision}</code>} />
          <Field label="Prompt hash (SHA-256)" value={<code>{inv.prompt_sha256}</code>} />
          <Field label="Output hash (SHA-256)" value={<code>{inv.output_sha256 ?? NOT_CAPTURED}</code>} />
          <Field label="Linked span" value={<code>{inv.span_id ?? NOT_CAPTURED}</code>} />
          <Field
            label="Neural access"
            value={inv.neural_access === "unavailable" ? "unavailable: no activations were captured" : "available"}
          />
        </dl>
      ))}
    </Section>
  );
}

function depthOf(span: SpanOut, byId: Map<string, SpanOut>): number {
  let depth = 0;
  for (let p = span.parent_span_id; p && byId.has(p) && depth < 50; depth++) {
    p = byId.get(p)!.parent_span_id;
  }
  return depth;
}

function Trace({ spans, traceId }: { spans: SpanOut[]; traceId: string | null }) {
  const byId = new Map(spans.map((s) => [s.span_id, s]));
  return (
    <Section title="Trace">
      <p>
        Trace ID: <code>{traceId ?? NOT_CAPTURED}</code>
      </p>
      {spans.length === 0 && <p>No spans recorded for this run.</p>}
      <ul className="spans">
        {spans.map((span) => {
          const ms = Date.parse(span.end_time) - Date.parse(span.start_time);
          return (
            <li key={span.span_id} style={{ marginLeft: `${depthOf(span, byId) * 1.5}rem` }}>
              <strong>{span.name}</strong> · {span.openinference_span_kind ?? "unknown kind"} ·{" "}
              {span.status_code === "ERROR" ? "✖ error" : "✔ ok"} · {ms} ms
            </li>
          );
        })}
      </ul>
    </Section>
  );
}

export function RunPage({ runId }: { runId: string | null }) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    fetchRun(runId, controller.signal).then(
      (run) => setState({ kind: "ready", run }),
      (err: unknown) => {
        if (controller.signal.aborted) return;
        const code = err instanceof ApiRequestError ? err.code : "UNREACHABLE";
        setState({ kind: "failed", code, message: err instanceof Error ? err.message : String(err) });
      },
    );
    return () => controller.abort();
  }, [runId]);

  return (
    <main>
      <h1>AI Black Box Lab: run</h1>
      {!runId && <p>Open this page with <code>?run=&lt;run_id&gt;</code> (the demo CLI prints the link).</p>}
      {runId && state.kind === "loading" && <p role="status">Loading run…</p>}
      {runId && state.kind === "failed" && (
        <p role="alert">
          {state.code === "RUN_NOT_FOUND" ? "Run not found." : "Could not load the run."} {state.message}
        </p>
      )}
      {runId && state.kind === "ready" && (
        <>
          <p>
            Run <code>{state.run.run_id}</code> · case {state.run.case_id} · retrieval fault: {state.run.retrieval_fault}
          </p>
          <Answer run={state.run} />
          <RetrievedDocument retrievals={state.run.retrievals} />
          <Invocations invocations={state.run.invocations} />
          <Trace spans={state.run.spans} traceId={state.run.trace_id} />
        </>
      )}
    </main>
  );
}

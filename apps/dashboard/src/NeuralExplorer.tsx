import { useEffect, useState } from "react";
import { fetchCapture, fetchMeasurements } from "./api";
import type { CaptureJobOut, InvocationOut, MeasurementOut } from "./types";

type State =
  | { kind: "loading" }
  | { kind: "ready"; measurements: MeasurementOut[]; job: CaptureJobOut | null }
  | { kind: "failed"; message: string };

// Layer x token-position grid of activation L2 norms. Measurements only: no interpretation.
export function NeuralExplorer({ invocation }: { invocation: InvocationOut }) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    fetchMeasurements(invocation.id, controller.signal)
      .then(async (measurements) => {
        const jobId = measurements[0]?.capture_job_id;
        const job = jobId ? await fetchCapture(jobId, controller.signal) : null;
        setState({ kind: "ready", measurements, job });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setState({ kind: "failed", message: err instanceof Error ? err.message : String(err) });
      });
    return () => controller.abort();
  }, [invocation.id]);

  if (state.kind === "loading") return <p role="status">Loading neural measurements…</p>;
  if (state.kind === "failed") return <p role="alert">✖ Could not load measurements: {state.message}</p>;

  const activations = state.measurements.filter((m) => m.kind === "activation");
  if (activations.length === 0) {
    return (
      <p>
        No activations captured yet (not captured). Queue one with{" "}
        <code>POST /v1/invocations/{invocation.id}/captures</code>, then run the worker.
      </p>
    );
  }
  const hooks = [...new Set(activations.map((m) => m.hook_name))].sort();
  const positions = [...new Set(activations.map((m) => m.token_position))].sort((a, b) => a - b);
  const norm = (hook: string, position: number) =>
    activations.find((m) => m.hook_name === hook && m.token_position === position)?.summary.l2_norm;
  const replay = state.job?.replay;

  return (
    <>
      <p>
        <span className="badge">measured</span> Model {invocation.model_id}, revision{" "}
        <code>{invocation.model_revision}</code>, float32. Replay:{" "}
        {replay
          ? `${String(replay.mode)}, ${String(replay.status)}, max logit difference vs unhooked ${String(replay.max_logit_diff)}, greedy output ${replay.greedy_consistent ? "✔ reproduced" : "✖ not reproduced"}`
          : "not recorded"}
      </p>
      <table>
        <caption>Activation L2 norm by hook and token position (hook point: residual stream)</caption>
        <thead>
          <tr>
            <th scope="col">Hook</th>
            {positions.map((p) => (
              <th scope="col" key={p}>
                pos {p}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {hooks.map((hook) => (
            <tr key={hook}>
              <th scope="row">{hook}</th>
              {positions.map((p) => {
                const value = norm(hook, p);
                return <td key={p}>{typeof value === "number" ? value.toFixed(2) : "not captured"}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p>These are measured activation sizes. They are not an explanation of the model's reasoning.</p>
    </>
  );
}

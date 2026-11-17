// Mirrors the /v1 response models (apps/api/blackbox_api/schemas.py). The checked-in OpenAPI
// snapshot (tests/contract/openapi_v1.json) is the contract: update both together.

export interface DocumentOut {
  id: string;
  doc_key: string;
  version: number;
  sha256: string;
  valid_from: string;
  valid_to: string | null; // null marks the current version
  artifact_id: string | null;
}

export interface RetrievalOut {
  rank: number;
  score: number;
  document: DocumentOut;
}

export interface InvocationOut {
  id: string;
  run_id: string;
  trace_id: string | null;
  span_id: string | null;
  model_id: string;
  model_revision: string;
  tokenizer_revision: string;
  prompt_sha256: string;
  output_sha256: string | null;
  sampling_config: Record<string, unknown>;
  neural_access: "available" | "unavailable";
  prompt_artifact_id: string | null; // null unless the run opted in to content capture
  created_at: string;
}

export interface ArtifactOut {
  id: string;
  sha256: string;
  mime_type: string;
  nbytes: number;
  category: string;
  parents: string[];
  verified: boolean; // re-hashed on this request
  integrity_error: string | null;
}

export interface MeasurementOut {
  id: string;
  capture_job_id: string;
  kind: "activation" | "logits";
  hook_name: string;
  layer: number | null;
  token_position: number;
  artifact_id: string; // verify with GET /v1/artifacts/{id}; bytes are not served
  dtype: string;
  shape: number[];
  summary: Record<string, number>;
  evidence_class: "measured";
}

export interface CaptureJobOut {
  id: string;
  invocation_id: string;
  status: "queued" | "running" | "completed" | "failed" | "blocked_resource";
  error_type: string | null;
  plan: { hooks: string[]; positions: number[]; include_logits: boolean };
  replay: Record<string, unknown> | null;
  created_at: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface SpanOut {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  name: string;
  openinference_span_kind: string | null;
  status_code: string;
  start_time: string;
  end_time: string;
}

export interface AnswerOut {
  captured: boolean;
  text: string | null;
  sha256: string | null;
}

export interface RunOut {
  run_id: string;
  task_id: string;
  case_id: string;
  retrieval_fault: string;
  trace_id: string | null;
  status: string;
  error_type: string | null;
  started_at: string;
  ended_at: string | null;
  answer: AnswerOut;
  expected_answer: string;
  is_correct: boolean | null;
  invocations: InvocationOut[];
  retrievals: RetrievalOut[];
  spans: SpanOut[];
}

export interface ApiErrorBody {
  error: { code: string; message: string; request_id: string };
}

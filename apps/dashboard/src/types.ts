// Mirrors the /v1 response models (apps/api/blackbox_api/schemas.py). The checked-in OpenAPI
// snapshot (tests/contract/openapi_v1.json) is the contract: update both together.

export interface DocumentOut {
  id: string;
  doc_key: string;
  version: number;
  sha256: string;
  valid_from: string;
  valid_to: string | null; // null marks the current version
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
  created_at: string;
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

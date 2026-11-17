import type { ApiErrorBody, RunOut } from "./types";

export class ApiRequestError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

export async function fetchRun(runId: string, signal: AbortSignal): Promise<RunOut> {
  const response = await fetch(`/v1/runs/${encodeURIComponent(runId)}`, { signal });
  if (response.ok) return (await response.json()) as RunOut;
  let body: ApiErrorBody | null = null;
  try {
    body = (await response.json()) as ApiErrorBody;
  } catch {
    // Not our API's error shape (for example the proxy target is down).
  }
  throw new ApiRequestError(
    body?.error.code ?? "UNREACHABLE",
    body?.error.message ?? `Request failed with HTTP ${response.status}`,
  );
}

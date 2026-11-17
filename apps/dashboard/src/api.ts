import type { ApiErrorBody, ArtifactOut, RunOut } from "./types";

export class ApiRequestError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function getJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (response.ok) return (await response.json()) as T;
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

export const fetchRun = (runId: string, signal: AbortSignal) =>
  getJson<RunOut>(`/v1/runs/${encodeURIComponent(runId)}`, signal);

export const fetchArtifact = (artifactId: string, signal: AbortSignal) =>
  getJson<ArtifactOut>(`/v1/artifacts/${encodeURIComponent(artifactId)}`, signal);

/**
 * Discern API Client
 */

export const API_BASE = "http://127.0.0.1:8000";

export interface Candidate {
  identity_id: number;
  name: string;
  similarity: number;
  rank: number;
  thumbnail_url?: string | null;
}

export interface MatchResponse {
  decision: "ACCEPTED" | "UNKNOWN";
  predicted_id: number | null;
  predicted_name: string | null;
  calibrated_confidence: number;
  raw_similarity: number;
  competitor_similarity: number;
  margin: number;
  reason_code: string;
  human_reason: string;
  top_candidates: Candidate[];
  inference_time_ms?: number;
  baseline_comparison?: {
    decision: "ACCEPTED" | "UNKNOWN";
    predicted_id: number | null;
    predicted_name: string | null;
    similarity: number;
    threshold: number;
    is_false_accept: boolean;
    explanation: string;
  };
  operating_point: {
    alpha: number;
    threshold_tau: number;
    margin_delta: number;
  };
}

export interface CrossDatasetItem {
  display_name: string;
  exists: boolean;
  description: string;
  download_url: string;
  status: string;
  metrics: {
    status: string;
    sample_count: number;
    enrolled_identities: number;
    unenrolled_identities: number;
    full_set: {
      baseline: { auroc: number; tar_1pct: number; tar_01pct: number; dir_1pct: number };
      discern: { auroc: number; tar_1pct: number; tar_01pct: number; dir_1pct: number };
    };
    lowvar_subset: {
      baseline: { auroc: number; tar_1pct: number; tar_01pct: number; dir_1pct: number };
      discern: { auroc: number; tar_1pct: number; tar_01pct: number; dir_1pct: number };
    };
  } | null;
}

export interface RobustnessItem {
  name: string;
  description: string;
  tar_rate: number;
  dir_rank1: number;
  false_accept_rate: number;
  average_margin: number;
  avg_confidence: number;
}

export interface DemoStep {
  step: number;
  title: string;
  category: string;
  probe_url: string;
  caption: string;
  expected: "ACCEPTED" | "UNKNOWN";
  discern_verdict: "ACCEPTED" | "UNKNOWN";
  baseline_verdict: "ACCEPTED" | "UNKNOWN";
  is_false_accept_prevented: boolean;
  confidence: number;
  similarity: number;
  competitor_similarity: number;
  margin: number;
  margin_delta: number;
  threshold_tau: number;
  human_reason: string;
  top_candidates: Candidate[];
}

export interface DemoScenarioResponse {
  status: string;
  scenario_name: string;
  steps: DemoStep[];
}

export interface GalleryIdentity {
  identity_id: number;
  name: string;
  num_samples: number;
  num_exemplars: number;
  adaptive_tau: number;
  nearest_lookalike_id: number | null;
  nearest_lookalike_sim: number;
  image_paths: string[];
  image_urls: string[];
}

export interface HeadlineMetrics {
  discern_lowvar_auroc: number;
  discern_lowvar_tar_1pct: number;
  discern_lowvar_tar_01pct: number;
  discern_lowvar_dir_1pct: number;
  baseline_lowvar_tar_1pct: number;
  relative_improvement_pct: number;
  cpu_latency_ms: number;
  parameters_million: number;
}

export interface AblationRow {
  component: string;
  key?: string;
  realized_far_1pct: number;
  realized_far_1pct_std?: number;
  realized_far_1pct_ci?: [number, number];
  full_tar_1pct: number;
  full_tar_1pct_std?: number;
  full_tar_1pct_ci?: [number, number];
  full_dir_1pct: number;
  full_dir_1pct_std?: number;
  full_dir_1pct_ci?: [number, number];
  full_auroc: number;
  full_auroc_std?: number;
  full_auroc_ci?: [number, number];
  p_value_tar_vs_baseline?: number;
  p_value_auroc_vs_baseline?: number;
  lowvar_realized_far_1pct?: number;
  lowvar_tar_1pct: number;
  lowvar_tar_1pct_std?: number;
  lowvar_tar_1pct_ci?: [number, number];
  lowvar_dir_1pct: number;
  lowvar_dir_1pct_std?: number;
  lowvar_dir_1pct_ci?: [number, number];
  lowvar_auroc: number;
  lowvar_auroc_std?: number;
  lowvar_auroc_ci?: [number, number];
  realized_far_01pct?: number;
  full_tar_01pct?: number;
  val_tar_1pct?: number;
}

export interface UniformStressTest {
  description: string;
  baseline: {
    realized_far: number;
    tar_1pct: number;
    tar_1pct_std: number;
    dir_1pct: number;
    auroc: number;
  };
  discern_default: {
    realized_far: number;
    tar_1pct: number;
    tar_1pct_std: number;
    dir_1pct: number;
    auroc: number;
  };
}

export interface LookAlikePair {
  identity_a: number;
  identity_b: number;
  clothing_similarity: number;
  is_same_cluster: boolean;
  image_url_a?: string;
  image_url_b?: string;
}

export const api = {
  async getGallery(): Promise<GalleryIdentity[]> {
    const res = await fetch(`${API_BASE}/gallery`);
    if (!res.ok) throw new Error("Failed to load gallery");
    return res.json();
  },

  async enrollIdentity(name: string, files: File[]): Promise<any> {
    const formData = new FormData();
    formData.append("name", name);
    for (const f of files) {
      formData.append("files", f);
    }
    const res = await fetch(`${API_BASE}/enroll`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Enrollment failed");
    }
    return res.json();
  },

  async deleteIdentity(id: number): Promise<any> {
    const res = await fetch(`${API_BASE}/identity/${id}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error("Failed to delete identity");
    return res.json();
  },

  async matchFile(file: File): Promise<MatchResponse> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/match`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Match request failed");
    }
    return res.json();
  },

  async matchBase64(base64: string): Promise<MatchResponse> {
    const res = await fetch(`${API_BASE}/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: base64 }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Webcam match failed");
    }
    return res.json();
  },

  async setOperatingPoint(alpha: number): Promise<any> {
    const res = await fetch(`${API_BASE}/operating-point`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alpha }),
    });
    if (!res.ok) throw new Error("Failed to update operating point");
    return res.json();
  },

  async getMetrics(): Promise<any> {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) throw new Error("Failed to fetch metrics");
    return res.json();
  },

  async getROC(): Promise<any> {
    const res = await fetch(`${API_BASE}/roc`);
    if (!res.ok) throw new Error("Failed to fetch ROC data");
    return res.json();
  },

  async getAblation(): Promise<any> {
    const res = await fetch(`${API_BASE}/ablation`);
    if (!res.ok) throw new Error("Failed to fetch ablation data");
    return res.json();
  },

  async getLookalikes(): Promise<any> {
    const res = await fetch(`${API_BASE}/lookalikes`);
    if (!res.ok) throw new Error("Failed to fetch lookalikes");
    return res.json();
  },

  async getCrossDataset(): Promise<{ status: string; datasets: Record<string, CrossDatasetItem> }> {
    const res = await fetch(`${API_BASE}/cross-dataset`);
    if (!res.ok) throw new Error("Failed to fetch cross-dataset results");
    return res.json();
  },

  async getRobustness(): Promise<{ status: string; perturbations: Record<string, RobustnessItem> }> {
    const res = await fetch(`${API_BASE}/robustness`);
    if (!res.ok) throw new Error("Failed to fetch robustness benchmarks");
    return res.json();
  },

  async runDemoScenario(): Promise<DemoScenarioResponse> {
    const res = await fetch(`${API_BASE}/demo/scenario`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Failed to execute live demo scenario");
    return res.json();
  },
};

import React, { useEffect, useState } from "react";
import { api, HeadlineMetrics, API_BASE } from "../api";
import { ShieldAlert, ArrowRight, Layers, Sliders, CheckCircle2, AlertTriangle, Cpu, Activity } from "lucide-react";

export const OverviewPage: React.FC<{ onNavigate: (page: any) => void }> = ({ onNavigate }) => {
  const [metrics, setMetrics] = useState<HeadlineMetrics | null>(null);
  const [analysisText, setAnalysisText] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getMetrics();
      if (res.status === "ready" && res.headline_metrics) {
        setMetrics(res.headline_metrics);
        setAnalysisText(res.analysis_paragraph || "");
      }
    } catch (err: any) {
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto py-6">
      {/* Title & Core Problem Paragraph */}
      <div className="border border-[#E5E7EB] rounded p-6 bg-white">
        <div className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-[#F9FAFB] text-[#111827] border border-[#E5E7EB] mb-3">
          Problem & Architecture
        </div>
        <h1 className="text-2xl font-bold text-[#111827] tracking-tight mb-3">
          Open-Set Person Re-Identification under Low Appearance Variance
        </h1>
        <p className="text-sm text-[#4B5563] leading-relaxed max-w-4xl">
          Standard person re-identification models depend heavily on dominant color cues (e.g. clothing color) and fail catastrophically when subjects share uniforms (security staff, hospital workers, warehouse personnel). Under open-set deployment, naive cosine similarity thresholds produce massive false accepts on look-alikes. <strong className="text-[#111827]">Discern</strong> resolves this via a four-stage pipeline: (1) horizontal multi-stripe pooling to isolate micro-features like badges and shoes, (2) gallery-adaptive covariance whitening to suppress shared uniform dimensions, (3) a dual-barrier decision rule requiring both per-identity adaptive similarity (s₁ ≥ τᵢ) and competitive identity margin (s₁ - s₂ ≥ δ), and (4) isotonic score calibration for rigorous target False Accept Rate (FAR) control.
        </p>
      </div>

      {/* Headline Metrics Cards from Real Results */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-[#111827] uppercase tracking-wider">
            Empirical Headline Benchmarks (From Real Evaluation)
          </h2>
          <span className="text-xs text-[#6B7280]">Dataset: Market-1501 Low-Variance Subset</span>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-28 bg-[#F9FAFB] border border-[#E5E7EB] rounded animate-pulse" />
            ))}
          </div>
        ) : error || !metrics ? (
          <div className="p-8 border border-[#E5E7EB] rounded text-center bg-[#F9FAFB]">
            <AlertTriangle className="w-6 h-6 text-[#D97706] mx-auto mb-2" />
            <p className="text-sm font-medium text-[#111827]">Run evaluation to populate</p>
            <p className="text-xs text-[#6B7280] mt-1">Execute scripts/evaluate.py to generate real metrics.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Card 1: LowVar TAR@1% */}
            <div className="p-5 border border-[#E5E7EB] rounded bg-white">
              <div className="text-xs font-medium text-[#6B7280] mb-1">TAR @ FAR=1.0% (LowVar)</div>
              <div className="flex items-baseline space-x-2">
                <span className="text-3xl font-bold text-[#111827]">
                  {(metrics.discern_lowvar_tar_1pct * 100).toFixed(1)}%
                </span>
                <span className="text-xs font-medium text-[#16A34A] bg-green-50 px-1.5 py-0.5 rounded border border-green-200">
                  +{metrics.relative_improvement_pct.toFixed(1)}% vs baseline
                </span>
              </div>
              <div className="text-xs text-[#6B7280] mt-2">
                Baseline plain cosine: {(metrics.baseline_lowvar_tar_1pct * 100).toFixed(1)}%
              </div>
            </div>

            {/* Card 2: DIR@1% */}
            <div className="p-5 border border-[#E5E7EB] rounded bg-white">
              <div className="text-xs font-medium text-[#6B7280] mb-1">DIR @ FAR=1.0% (Identification)</div>
              <div className="flex items-baseline space-x-2">
                <span className="text-3xl font-bold text-[#111827]">
                  {(metrics.discern_lowvar_dir_1pct * 100).toFixed(1)}%
                </span>
                <span className="text-xs text-[#6B7280]">True Pos ID</span>
              </div>
              <div className="text-xs text-[#6B7280] mt-2">Correct detection & identity attribution</div>
            </div>

            {/* Card 3: Inference Latency */}
            <div className="p-5 border border-[#E5E7EB] rounded bg-white">
              <div className="text-xs font-medium text-[#6B7280] mb-1">Inference Latency (CPU)</div>
              <div className="flex items-baseline space-x-2">
                <span className="text-3xl font-bold text-[#111827]">
                  {metrics.cpu_latency_ms.toFixed(1)}
                </span>
                <span className="text-sm font-medium text-[#6B7280]">ms / image</span>
              </div>
              <div className="text-xs text-[#6B7280] mt-2">Throughput: ~{(1000 / metrics.cpu_latency_ms).toFixed(1)} FPS</div>
            </div>

            {/* Card 4: Model Footprint */}
            <div className="p-5 border border-[#E5E7EB] rounded bg-white">
              <div className="text-xs font-medium text-[#6B7280] mb-1">Compact Model Footprint</div>
              <div className="flex items-baseline space-x-2">
                <span className="text-3xl font-bold text-[#111827]">
                  {metrics.parameters_million.toFixed(2)}M
                </span>
                <span className="text-sm font-medium text-[#6B7280]">weights</span>
              </div>
              <div className="text-xs text-[#6B7280] mt-2">OSNet multi-scale backbone</div>
            </div>
          </div>
        )}
      </div>

      {/* Plain Language Analysis Paragraph from Real Run */}
      {analysisText && (
        <div className="p-5 border border-[#E5E7EB] rounded bg-[#F9FAFB]">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="w-4 h-4 text-[#111827]" />
            <h3 className="text-xs font-semibold uppercase text-[#111827] tracking-wider">
              Automated Empirical Analysis Summary
            </h3>
          </div>
          <p className="text-xs text-[#374151] leading-relaxed italic">
            "{analysisText}"
          </p>
        </div>
      )}

      {/* Simple, Clean Pipeline Diagram */}
      <div className="border border-[#E5E7EB] rounded p-6 bg-white">
        <h2 className="text-sm font-semibold text-[#111827] uppercase tracking-wider mb-4">
          Discern Decision Architecture Flow
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-center">
          {/* Step 1 */}
          <div className="p-4 border border-[#E5E7EB] rounded bg-[#F9FAFB] text-center">
            <div className="w-8 h-8 rounded bg-white border border-[#E5E7EB] flex items-center justify-center mx-auto mb-2 text-xs font-bold">
              1
            </div>
            <div className="text-xs font-semibold text-[#111827]">Input Image</div>
            <div className="text-[11px] text-[#6B7280] mt-1">256×128 Re-ID Crop</div>
          </div>

          {/* Step 2 */}
          <div className="p-4 border border-[#E5E7EB] rounded bg-[#F9FAFB] text-center">
            <div className="w-8 h-8 rounded bg-white border border-[#E5E7EB] flex items-center justify-center mx-auto mb-2 text-xs font-bold">
              2
            </div>
            <div className="text-xs font-semibold text-[#111827]">Stripe Features</div>
            <div className="text-[11px] text-[#6B7280] mt-1">Global + 3 Stripes (512-d)</div>
          </div>

          {/* Step 3 */}
          <div className="p-4 border border-[#E5E7EB] rounded bg-[#F9FAFB] text-center">
            <div className="w-8 h-8 rounded bg-white border border-[#E5E7EB] flex items-center justify-center mx-auto mb-2 text-xs font-bold">
              3
            </div>
            <div className="text-xs font-semibold text-[#111827]">Adaptive Whitening</div>
            <div className="text-[11px] text-[#6B7280] mt-1">Suppress Uniform Covariance</div>
          </div>

          {/* Step 4 */}
          <div className="p-4 border border-[#E5E7EB] rounded bg-[#F9FAFB] text-center">
            <div className="w-8 h-8 rounded bg-white border border-[#E5E7EB] flex items-center justify-center mx-auto mb-2 text-xs font-bold">
              4
            </div>
            <div className="text-xs font-semibold text-[#111827]">Dual-Barrier Test</div>
            <div className="text-[11px] text-[#6B7280] mt-1">s₁ ≥ τᵢ & (s₁ - s₂) ≥ δ</div>
          </div>

          {/* Step 5 */}
          <div className="p-4 border border-[#E5E7EB] rounded bg-[#F9FAFB] text-center">
            <div className="w-8 h-8 rounded bg-white border border-[#E5E7EB] flex items-center justify-center mx-auto mb-2 text-xs font-bold">
              5
            </div>
            <div className="text-xs font-semibold text-[#111827]">Calibrated Output</div>
            <div className="text-[11px] text-[#6B7280] mt-1">ACCEPTED or UNKNOWN</div>
          </div>
        </div>

        {/* Quick action buttons */}
        <div className="mt-6 pt-5 border-t border-[#E5E7EB] flex justify-between items-center">
          <span className="text-xs text-[#6B7280]">
            Ready to test open-set person recognition?
          </span>
          <div className="space-x-2">
            <button
              onClick={() => onNavigate("match")}
              className="px-4 py-2 bg-[#111827] text-white text-xs font-medium rounded hover:bg-black transition-colors"
            >
              Go to Match & Verify →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

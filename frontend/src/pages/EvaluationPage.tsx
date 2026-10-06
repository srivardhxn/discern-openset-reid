import React, { useEffect, useState } from "react";
import { api, API_BASE } from "../api";
import { BarChart3, AlertTriangle, Award, Cpu } from "lucide-react";

// ---------------------------------------------------------------------------
// Types matching eval_report.json from the model bundle
// ---------------------------------------------------------------------------
interface ClosedSet {
  mAP: number;
  R1: number;
  R5: number;
  R10: number;
}

interface AblationEntry {
  [metric: string]: [number, number]; // [mean, std]
}

interface TrainingAblationEntry {
  mAP: number;
  R1: number;
  far90_full: number;
  far90_lv90: number;
  tar1_lv90: number;
}

interface OpPointEntry {
  threshold: number;
  target_far: number;
  far: number;
  far_lookalike_lv90: number;
  tar: number;
}

interface EvalReport {
  closed_set_market1501: ClosedSet;
  selected_weights: string;
  best_aggregation: string;
  calibration_heldout: { ECE: number; Brier: number };
  open_set_ablation: Record<string, AblationEntry>;
  training_ablation: Record<string, TrainingAblationEntry>;
  operating_points: Record<string, Record<string, OpPointEntry>>;
  hardware: { gpus: string[]; epochs: number; batch: number; input_hw: [number, number] };
}

interface MetricImages {
  roc_ablation?: string;
  calibration?: string;
  training_curves?: string;
  lookalike_pairs?: string;
}

// ---------------------------------------------------------------------------
// EvaluationPage
// ---------------------------------------------------------------------------
export const EvaluationPage: React.FC = () => {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [images, setImages] = useState<MetricImages>({});
  const [ablationInf, setAblationInf] = useState<string>("");
  const [ablationTrain, setAblationTrain] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { loadAll(); }, []);

  const loadAll = async () => {
    try {
      setLoading(true);
      setError(null);
      const [rpt, imgs, inf, trn] = await Promise.all([
        api.getMetricsNew(),
        api.getMetricImages(),
        fetch(`${API_BASE}/api/metrics/ablation-inference`).then((r) => r.json()),
        fetch(`${API_BASE}/api/metrics/ablation-training`).then((r) => r.json()),
      ]);
      setReport(rpt);
      setImages(imgs);
      setAblationInf(inf.content || "");
      setAblationTrain(trn.content || "");
    } catch (err: any) {
      setError(err.message || "Failed to load evaluation data");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-[1200px] mx-auto py-12 text-center font-sans">
        <div className="w-8 h-8 border-2 border-[#0A0A0A] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-xs text-[#6B7280]">Loading evaluation metrics from model bundle…</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-[1200px] mx-auto py-12 font-sans">
        <div className="p-8 border border-[#E5E7EB] rounded-lg text-center bg-[#FAFAFA]">
          <AlertTriangle className="w-8 h-8 text-[#D97706] mx-auto mb-2" />
          <h2 className="text-sm font-semibold text-[#0A0A0A]">Evaluation Data Missing</h2>
          <p className="text-xs text-[#6B7280] mt-1">
            {error || "reports/eval_report.json not found in model bundle."}
          </p>
        </div>
      </div>
    );
  }

  const cs = report.closed_set_market1501;
  const cal = report.calibration_heldout;
  const hw = report.hardware;

  // Pull the best (final) row from inference ablation
  const ablationSteps = Object.entries(report.open_set_ablation);
  const finalStep = ablationSteps[ablationSteps.length - 1];
  const finalData = finalStep?.[1] ?? {};

  // s1_margin_z operating points (the preferred calibrator)
  const opPoints = report.operating_points?.s1_margin_z ?? {};

  return (
    <div className="space-y-8 max-w-[1200px] mx-auto py-6 font-sans">

      {/* ── Page Header ──────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#E5E7EB] gap-3">
        <div>
          <h1 className="text-xl font-bold text-[#0A0A0A] tracking-tight">
            Evaluation & Benchmarks
          </h1>
          <p className="text-xs text-[#6B7280] mt-0.5">
            Discern-R50-BNNeck · ONNX bundle · Market-1501 open-set protocol (10 random 50/50 splits)
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-[#6B7280]">
          <span className="px-2.5 py-1 bg-[#FAFAFA] border border-[#E5E7EB] rounded">
            Weights: {report.selected_weights} · Aggregation: {report.best_aggregation}
          </span>
          <span className="px-2.5 py-1 bg-[#FAFAFA] border border-[#E5E7EB] rounded">
            GPU: {hw.gpus?.join(", ")} · {hw.epochs} epochs
          </span>
        </div>
      </div>

      {/* ── Headline Metrics ─────────────────────────────────────────────── */}
      <div>
        <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider mb-3 flex items-center">
          <Award className="w-3.5 h-3.5 mr-1.5" /> Headline Metrics
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: "Closed-Set mAP", value: cs.mAP.toFixed(1) + "%", sub: "Market-1501" },
            { label: "Rank-1 Accuracy", value: cs.R1.toFixed(1) + "%",  sub: "Closed-set" },
            { label: "Calibration ECE",  value: cal.ECE.toFixed(4),       sub: "↓ better" },
            { label: "Brier Score",      value: cal.Brier.toFixed(4),     sub: "↓ better" },
          ].map((m) => (
            <div key={m.label} className="border border-[#E5E7EB] rounded-lg p-4 bg-white text-center">
              <div className="text-[11px] text-[#6B7280] font-medium mb-1">{m.label}</div>
              <div className="text-2xl font-bold font-mono text-[#0A0A0A]">{m.value}</div>
              <div className="text-[10px] text-[#6B7280] mt-0.5">{m.sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Operating-Point Table ────────────────────────────────────────── */}
      <div>
        <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider mb-3 flex items-center">
          <BarChart3 className="w-3.5 h-3.5 mr-1.5" /> Operating Points — Calibrator: s1_margin_z
        </h2>
        <div className="border border-[#E5E7EB] rounded-lg overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-[#FAFAFA] border-b border-[#E5E7EB]">
                {["Operating Point", "Threshold τ", "Target FAR", "Actual FAR", "FAR look-alike (LV90)", "TAR"].map((h) => (
                  <th key={h} className="px-3 py-2.5 text-left font-semibold text-[#6B7280] text-[11px]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E7EB]">
              {Object.entries(opPoints).map(([opName, op]) => (
                <tr key={opName} className="hover:bg-[#FAFAFA]">
                  <td className="px-3 py-2.5 font-semibold capitalize text-[#0A0A0A]">{opName}</td>
                  <td className="px-3 py-2.5 font-mono text-[#0A0A0A]">{op.threshold.toFixed(4)}</td>
                  <td className="px-3 py-2.5 font-mono text-[#6B7280]">{(op.target_far * 100).toFixed(2)}%</td>
                  <td className="px-3 py-2.5 font-mono text-[#0A0A0A]">{(op.far * 100).toFixed(3)}%</td>
                  <td className="px-3 py-2.5 font-mono text-[#D97706]">{(op.far_lookalike_lv90 * 100).toFixed(3)}%</td>
                  <td className="px-3 py-2.5 font-mono font-bold text-[#16A34A]">{(op.tar * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[10px] text-[#6B7280] mt-1.5 font-mono">
          Protocol: Market-1501 test split, 750 identities, 50% enrolled / 50% unknown, 10 random seeds.
          LV90 = probes whose nearest enrolled colour-twin is in the top 10% of HSV similarity.
        </p>
      </div>

      {/* ── ROC & Ablation PNG images ─────────────────────────────────────── */}
      {(images.roc_ablation || images.calibration || images.training_curves || images.lookalike_pairs) && (
        <div>
          <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider mb-3 flex items-center">
            <BarChart3 className="w-3.5 h-3.5 mr-1.5" /> Report Charts
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {images.roc_ablation && (
              <div className="border border-[#E5E7EB] rounded-lg overflow-hidden bg-white">
                <div className="px-4 py-2.5 border-b border-[#E5E7EB] bg-[#FAFAFA]">
                  <span className="text-[11px] font-semibold text-[#0A0A0A]">Open-Set ROC & Inference Ablation</span>
                </div>
                <img
                  src={`${API_BASE}${images.roc_ablation}`}
                  alt="ROC ablation"
                  className="w-full object-contain"
                />
              </div>
            )}
            {images.calibration && (
              <div className="border border-[#E5E7EB] rounded-lg overflow-hidden bg-white">
                <div className="px-4 py-2.5 border-b border-[#E5E7EB] bg-[#FAFAFA]">
                  <span className="text-[11px] font-semibold text-[#0A0A0A]">Calibration Reliability Diagram</span>
                </div>
                <img
                  src={`${API_BASE}${images.calibration}`}
                  alt="Calibration reliability"
                  className="w-full object-contain"
                />
              </div>
            )}
            {images.training_curves && (
              <div className="border border-[#E5E7EB] rounded-lg overflow-hidden bg-white">
                <div className="px-4 py-2.5 border-b border-[#E5E7EB] bg-[#FAFAFA]">
                  <span className="text-[11px] font-semibold text-[#0A0A0A]">Training Curves</span>
                </div>
                <img
                  src={`${API_BASE}${images.training_curves}`}
                  alt="Training curves"
                  className="w-full object-contain"
                />
              </div>
            )}
            {images.lookalike_pairs && (
              <div className="border border-[#E5E7EB] rounded-lg overflow-hidden bg-white">
                <div className="px-4 py-2.5 border-b border-[#E5E7EB] bg-[#FAFAFA]">
                  <span className="text-[11px] font-semibold text-[#0A0A0A]">Look-Alike Pairs (LV90)</span>
                </div>
                <img
                  src={`${API_BASE}${images.lookalike_pairs}`}
                  alt="Lookalike pairs"
                  className="w-full object-contain"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Inference Ablation Table ─────────────────────────────────────── */}
      <div>
        <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider mb-3">
          Inference Ablation (open_set_ablation)
        </h2>
        <div className="border border-[#E5E7EB] rounded-lg overflow-auto">
          <table className="w-full text-xs whitespace-nowrap">
            <thead>
              <tr className="bg-[#FAFAFA] border-b border-[#E5E7EB]">
                <th className="px-3 py-2.5 text-left font-semibold text-[#6B7280] text-[11px]">Step</th>
                {["full|TAR@FAR1%", "full|AUROC", "LV90|TAR@FAR1%", "LV90|AUROC"].map((h) => (
                  <th key={h} className="px-3 py-2.5 text-right font-semibold text-[#6B7280] text-[11px]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E7EB]">
              {ablationSteps.map(([stepName, stepData], idx) => (
                <tr key={idx} className={idx === ablationSteps.length - 1 ? "bg-green-50/40 font-semibold" : "hover:bg-[#FAFAFA]"}>
                  <td className="px-3 py-2.5 text-[#0A0A0A] max-w-xs truncate" title={stepName}>{stepName}</td>
                  {["full|TAR@FAR1%", "full|AUROC", "LV90|TAR@FAR1%", "LV90|AUROC"].map((key) => {
                    const val = (stepData as any)[key];
                    const mean = Array.isArray(val) ? val[0] : val;
                    const std  = Array.isArray(val) ? val[1] : null;
                    return (
                      <td key={key} className="px-3 py-2.5 text-right font-mono text-[#0A0A0A]">
                        {mean != null ? (mean * 100).toFixed(1) + "%" : "—"}
                        {std != null && <span className="text-[#6B7280] text-[10px]"> ±{(std * 100).toFixed(1)}</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Training Ablation Table ──────────────────────────────────────── */}
      <div>
        <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider mb-3">
          Training Recipe Ablation (training_ablation)
        </h2>
        <div className="border border-[#E5E7EB] rounded-lg overflow-auto">
          <table className="w-full text-xs whitespace-nowrap">
            <thead>
              <tr className="bg-[#FAFAFA] border-b border-[#E5E7EB]">
                <th className="px-3 py-2.5 text-left font-semibold text-[#6B7280] text-[11px]">Training Step</th>
                {["mAP", "R1", "FAR90 (full)", "FAR90 (LV90)", "TAR@1% (LV90)"].map((h) => (
                  <th key={h} className="px-3 py-2.5 text-right font-semibold text-[#6B7280] text-[11px]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E7EB]">
              {Object.entries(report.training_ablation).map(([stepName, s], idx, arr) => (
                <tr key={idx} className={idx === arr.length - 1 ? "bg-green-50/40 font-semibold" : "hover:bg-[#FAFAFA]"}>
                  <td className="px-3 py-2.5 text-[#0A0A0A] max-w-xs truncate" title={stepName}>{stepName}</td>
                  <td className="px-3 py-2.5 text-right font-mono text-[#0A0A0A]">{s.mAP.toFixed(1)}%</td>
                  <td className="px-3 py-2.5 text-right font-mono text-[#0A0A0A]">{s.R1.toFixed(1)}%</td>
                  <td className="px-3 py-2.5 text-right font-mono text-[#D97706]">{s.far90_full.toFixed(1)}%</td>
                  <td className="px-3 py-2.5 text-right font-mono text-[#D97706]">{s.far90_lv90.toFixed(1)}%</td>
                  <td className="px-3 py-2.5 text-right font-mono font-bold text-[#16A34A]">{s.tar1_lv90.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Ablation markdown (text summary) ─────────────────────────────── */}
      {(ablationInf || ablationTrain) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {ablationInf && (
            <div className="border border-[#E5E7EB] rounded-lg p-4 bg-[#FAFAFA]">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[#0A0A0A] mb-2">
                Inference Ablation Notes
              </h3>
              <pre className="text-[10px] text-[#6B7280] whitespace-pre-wrap font-mono leading-relaxed">
                {ablationInf}
              </pre>
            </div>
          )}
          {ablationTrain && (
            <div className="border border-[#E5E7EB] rounded-lg p-4 bg-[#FAFAFA]">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[#0A0A0A] mb-2">
                Training Ablation Notes
              </h3>
              <pre className="text-[10px] text-[#6B7280] whitespace-pre-wrap font-mono leading-relaxed">
                {ablationTrain}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* ── Hardware info ─────────────────────────────────────────────────── */}
      <div className="border border-[#E5E7EB] rounded-lg p-4 bg-white">
        <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider mb-3 flex items-center">
          <Cpu className="w-3.5 h-3.5 mr-1.5" /> Hardware & Training Config
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
          <div><span className="text-[#6B7280]">GPUs: </span><span className="text-[#0A0A0A]">{hw.gpus?.join(", ")}</span></div>
          <div><span className="text-[#6B7280]">Epochs: </span><span className="text-[#0A0A0A]">{hw.epochs}</span></div>
          <div><span className="text-[#6B7280]">Batch: </span><span className="text-[#0A0A0A]">{hw.batch}</span></div>
          <div><span className="text-[#6B7280]">Input HW: </span><span className="text-[#0A0A0A]">{hw.input_hw?.[0]}×{hw.input_hw?.[1]}</span></div>
        </div>
      </div>

    </div>
  );
};

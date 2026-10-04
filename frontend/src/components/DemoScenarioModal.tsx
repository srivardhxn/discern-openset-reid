import React, { useState, useEffect } from "react";
import { api, DemoStep, DemoScenarioResponse, API_BASE } from "../api";
import { HorizontalSimilarityBar } from "./HorizontalSimilarityBar";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ArrowRight,
  ArrowLeft,
  RotateCcw,
  X,
  Play,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";

interface DemoScenarioModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DemoScenarioModal: React.FC<DemoScenarioModalProps> = ({ isOpen, onClose }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [scenarioData, setScenarioData] = useState<DemoScenarioResponse | null>(null);
  const [currentStepIdx, setCurrentStepIdx] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadScenario();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") handleNext();
      if (e.key === "ArrowLeft") handlePrev();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, currentStepIdx, scenarioData]);

  const loadScenario = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.runDemoScenario();
      setScenarioData(res);
      setCurrentStepIdx(0);
    } catch (err: any) {
      setError(err.message || "Failed to execute demo scenario");
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    if (scenarioData && currentStepIdx < scenarioData.steps.length - 1) {
      setCurrentStepIdx((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    if (currentStepIdx > 0) {
      setCurrentStepIdx((prev) => prev - 1);
    }
  };

  if (!isOpen) return null;

  const currentStep: DemoStep | undefined = scenarioData?.steps[currentStepIdx];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="relative w-full max-w-4xl bg-white border border-[#E5E7EB] rounded-lg shadow-xl overflow-hidden font-sans">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#E5E7EB] bg-white">
          <div className="flex items-center space-x-3">
            <div className="w-7 h-7 rounded bg-[#0A0A0A] text-white flex items-center justify-center">
              <Play className="w-3.5 h-3.5 fill-current" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-[#0A0A0A]">
                Live Judge Demo Scenario
              </h2>
              <p className="text-xs text-[#6B7280]">
                3-Step Open-Set Audit using Real OSNet Embeddings & Market-1501 Benchmark Probes
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-[#6B7280] hover:text-[#0A0A0A] p-1.5 rounded hover:bg-[#FAFAFA] transition-colors focus:outline-none focus:ring-1 focus:ring-[#0A0A0A]"
            aria-label="Close modal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Stepper Progress Bar */}
        <div className="grid grid-cols-3 border-b border-[#E5E7EB] bg-[#FAFAFA] text-xs">
          {scenarioData?.steps.map((s, idx) => {
            const isActive = idx === currentStepIdx;
            const isCompleted = idx < currentStepIdx;
            return (
              <button
                key={idx}
                onClick={() => setCurrentStepIdx(idx)}
                className={`py-3 px-4 text-left border-r border-[#E5E7EB] last:border-r-0 transition-colors ${
                  isActive
                    ? "bg-white font-semibold text-[#0A0A0A] border-b-2 border-b-[#0A0A0A]"
                    : isCompleted
                    ? "text-[#16A34A] hover:bg-[#F3F4F6]"
                    : "text-[#6B7280] hover:bg-[#F3F4F6]"
                }`}
              >
                <div className="flex items-center space-x-2">
                  <span
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono ${
                      isActive
                        ? "bg-[#0A0A0A] text-white"
                        : isCompleted
                        ? "bg-[#16A34A] text-white"
                        : "bg-[#E5E7EB] text-[#6B7280]"
                    }`}
                  >
                    {idx + 1}
                  </span>
                  <span className="truncate">{s.title.split(":")[1]?.trim() || s.title}</span>
                </div>
              </button>
            );
          }) || (
            <div className="col-span-3 py-3 px-4 text-[#6B7280]">Loading benchmark probes...</div>
          )}
        </div>

        {/* Modal Body */}
        <div className="p-6">
          {loading ? (
            <div className="py-16 text-center space-y-3">
              <div className="w-8 h-8 border-2 border-[#0A0A0A] border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-xs text-[#6B7280]">Executing neural inference across gallery prototypes...</p>
            </div>
          ) : error ? (
            <div className="p-6 border border-red-200 bg-red-50 text-red-700 text-xs rounded text-center">
              <AlertTriangle className="w-6 h-6 mx-auto mb-2 text-[#DC2626]" />
              <p className="font-semibold">{error}</p>
              <button
                onClick={loadScenario}
                className="mt-3 px-3 py-1.5 bg-white border border-red-300 text-red-800 rounded font-medium"
              >
                Retry Scenario
              </button>
            </div>
          ) : currentStep ? (
            <div className="space-y-6">
              {/* Step Banner & Decision */}
              <div
                className={`p-4 rounded border ${
                  currentStep.discern_verdict === "ACCEPTED"
                    ? "bg-green-50/50 border-green-200"
                    : currentStep.is_false_accept_prevented
                    ? "bg-amber-50/50 border-amber-200"
                    : "bg-[#FAFAFA] border-[#E5E7EB]"
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center space-x-3">
                    {currentStep.discern_verdict === "ACCEPTED" ? (
                      <CheckCircle2 className="w-8 h-8 text-[#16A34A] flex-shrink-0" />
                    ) : (
                      <ShieldAlert className="w-8 h-8 text-[#DC2626] flex-shrink-0" />
                    )}
                    <div>
                      <div className="text-[10px] uppercase font-semibold text-[#6B7280] tracking-wider">
                        {currentStep.title}
                      </div>
                      <div className="text-xl font-bold text-[#0A0A0A]">
                        {currentStep.discern_verdict === "ACCEPTED"
                          ? `ACCEPTED as ${currentStep.top_candidates[0]?.name}`
                          : "UNKNOWN (Refused Re-ID)"}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4 text-right">
                    <div>
                      <span className="text-[10px] text-[#6B7280] uppercase block">Confidence</span>
                      <span className="text-xl font-mono tabular-nums font-bold text-[#0A0A0A]">
                        {(currentStep.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    {currentStep.is_false_accept_prevented && (
                      <div className="bg-[#D97706]/10 border border-[#D97706] text-[#D97706] px-2.5 py-1 rounded text-xs font-semibold">
                        False Accept Prevented!
                      </div>
                    )}
                  </div>
                </div>

                {/* Plain English Reason */}
                <div className="mt-3 pt-3 border-t border-black/5 text-xs text-[#0A0A0A] font-medium leading-relaxed">
                  <strong>Decision Rationale:</strong> {currentStep.human_reason}
                </div>
              </div>

              {/* Side-by-Side Images & Top Candidates */}
              <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                {/* Query Probe Image */}
                <div className="md:col-span-4 border border-[#E5E7EB] rounded p-3 bg-[#FAFAFA] text-center">
                  <span className="text-[11px] font-semibold text-[#6B7280] block mb-2 uppercase">
                    Query Probe Image
                  </span>
                  <img
                    src={`${API_BASE}${currentStep.probe_url}`}
                    alt="Probe image"
                    className="w-24 h-44 object-cover rounded mx-auto border border-[#E5E7EB] bg-white"
                  />
                  <span className="text-[10px] text-[#6B7280] font-mono mt-1 block truncate">
                    {currentStep.category}
                  </span>
                </div>

                {/* Matched Gallery Candidates */}
                <div className="md:col-span-8 space-y-3">
                  <span className="text-[11px] font-semibold text-[#6B7280] uppercase block">
                    Enrolled Gallery Candidates Evaluated
                  </span>
                  <div className="grid grid-cols-2 gap-3">
                    {currentStep.top_candidates.map((cand) => (
                      <div
                        key={cand.identity_id}
                        className="p-3 border border-[#E5E7EB] rounded bg-white flex items-center space-x-3"
                      >
                        {cand.thumbnail_url ? (
                          <img
                            src={`${API_BASE}${cand.thumbnail_url}`}
                            alt={cand.name}
                            className="w-12 h-20 object-cover rounded border border-[#E5E7EB]"
                          />
                        ) : (
                          <div className="w-12 h-20 bg-[#FAFAFA] border border-[#E5E7EB] rounded flex items-center justify-center text-[10px] text-[#6B7280]">
                            #{cand.rank}
                          </div>
                        )}
                        <div className="flex-1 min-w-0">
                          <span className="text-xs font-semibold text-[#0A0A0A] block truncate">
                            {cand.name}
                          </span>
                          <span className="text-[11px] font-mono text-[#6B7280] block">
                            Rank {cand.rank}
                          </span>
                          <span className="text-xs font-mono font-bold text-[#0A0A0A] mt-1 block">
                            {(cand.similarity * 100).toFixed(1)}% sim
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Horizontal Similarity Bar */}
                  <div className="border border-[#E5E7EB] rounded p-3 bg-white">
                    <HorizontalSimilarityBar
                      top1Similarity={currentStep.similarity}
                      top2Similarity={currentStep.competitor_similarity}
                      thresholdTau={currentStep.threshold_tau}
                      marginDelta={currentStep.margin_delta}
                      decision={currentStep.discern_verdict}
                      top1Name={currentStep.top_candidates[0]?.name || "Top 1"}
                      top2Name={currentStep.top_candidates[1]?.name || "Top 2"}
                    />
                  </div>
                </div>
              </div>

              {/* Compare Toggle Box (Baseline vs Discern) */}
              <div className="border border-[#E5E7EB] rounded p-4 bg-[#FAFAFA] space-y-3">
                <span className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider block">
                  Algorithm Comparison: Plain Cosine vs Discern
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  {/* Baseline Card */}
                  <div className="p-3 border border-[#E5E7EB] rounded bg-white">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-semibold text-[#6B7280]">Plain Cosine Baseline</span>
                      <span
                        className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                          currentStep.baseline_verdict === "ACCEPTED"
                            ? currentStep.is_false_accept_prevented
                              ? "bg-red-50 text-[#DC2626] border border-red-200"
                              : "bg-green-50 text-[#16A34A] border border-green-200"
                            : "bg-[#FAFAFA] text-[#6B7280] border border-[#E5E7EB]"
                        }`}
                      >
                        {currentStep.baseline_verdict}
                        {currentStep.is_false_accept_prevented && " (FALSE ACCEPT!)"}
                      </span>
                    </div>
                    <p className="text-[11px] text-[#6B7280] leading-relaxed">
                      {currentStep.is_false_accept_prevented
                        ? `Matches similarity ${(currentStep.similarity * 100).toFixed(1)}% >= 70.0% fixed threshold, erroneously admitting the look-alike impostor.`
                        : `Evaluated at fixed threshold 70.0% cosine similarity.`}
                    </p>
                  </div>

                  {/* Discern Card */}
                  <div className="p-3 border border-[#E5E7EB] rounded bg-white">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-semibold text-[#0A0A0A]">Discern (Dual-Barrier)</span>
                      <span
                        className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                          currentStep.discern_verdict === "ACCEPTED"
                            ? "bg-green-50 text-[#16A34A] border border-green-200"
                            : "bg-[#FAFAFA] text-[#0A0A0A] border border-[#E5E7EB]"
                        }`}
                      >
                        {currentStep.discern_verdict} (PROTECTED)
                      </span>
                    </div>
                    <p className="text-[11px] text-[#6B7280] leading-relaxed">
                      {currentStep.is_false_accept_prevented
                        ? `Whitens gallery covariance and requires margin Δ >= ${(currentStep.margin_delta * 100).toFixed(1)}%. Since margin is only ${(currentStep.margin * 100).toFixed(1)}%, impostor is safely refused.`
                        : `Enforces calibrated FAR control via isotonic mapping and competitive margin delta.`}
                    </p>
                  </div>
                </div>
                {/* Caption */}
                <p className="text-xs text-[#0A0A0A] italic bg-white p-2.5 rounded border border-[#E5E7EB]">
                  <strong>Judge Summary:</strong> {currentStep.caption}
                </p>
              </div>
            </div>
          ) : null}
        </div>

        {/* Modal Footer Controls */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-[#E5E7EB] bg-white">
          <button
            onClick={loadScenario}
            disabled={loading}
            className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-[#0A0A0A] bg-white border border-[#E5E7EB] rounded hover:bg-[#FAFAFA] transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
            Restart Demo
          </button>

          <div className="flex items-center space-x-2">
            <button
              onClick={handlePrev}
              disabled={currentStepIdx === 0 || loading}
              className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-[#0A0A0A] bg-white border border-[#E5E7EB] rounded hover:bg-[#FAFAFA] disabled:opacity-40 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5 mr-1.5" />
              Previous
            </button>

            {scenarioData && currentStepIdx < scenarioData.steps.length - 1 ? (
              <button
                onClick={handleNext}
                disabled={loading}
                className="inline-flex items-center px-4 py-1.5 text-xs font-medium text-white bg-[#0A0A0A] rounded hover:bg-black transition-colors"
              >
                Next Step
                <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
              </button>
            ) : (
              <button
                onClick={onClose}
                className="inline-flex items-center px-4 py-1.5 text-xs font-medium text-white bg-[#0A0A0A] rounded hover:bg-black transition-colors"
              >
                Finish Demo
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

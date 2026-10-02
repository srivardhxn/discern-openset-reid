import React, { useState, useRef, useEffect } from "react";
import { api, MatchResponse, API_BASE } from "../api";
import { HorizontalSimilarityBar } from "../components/HorizontalSimilarityBar";
import { DemoScenarioModal } from "../components/DemoScenarioModal";
import {
  UploadCloud,
  Camera,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Sliders,
  RefreshCw,
  ArrowRight,
  ShieldCheck,
  ShieldAlert,
  Play,
  Clock,
  ToggleLeft,
  ToggleRight,
  Users,
} from "lucide-react";

export const MatchPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [matchResult, setMatchResult] = useState<MatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // False Accept Rate (Alpha) Slider
  const [alpha, setAlpha] = useState<number>(0.01);
  const [alphaUpdating, setAlphaUpdating] = useState<boolean>(false);

  // Compare Toggle: Baseline Cosine vs Discern
  const [showComparison, setShowComparison] = useState<boolean>(true);

  // Demo Scenario Modal
  const [demoModalOpen, setDemoModalOpen] = useState<boolean>(false);

  // Webcam state
  const [useWebcam, setUseWebcam] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  const startWebcam = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 480, height: 640 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setUseWebcam(true);
    } catch (err: any) {
      setError("Unable to access webcam: " + (err.message || "Permission denied"));
      setUseWebcam(false);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setUseWebcam(false);
  };

  const captureWebcamFrame = async () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 300;
    canvas.height = videoRef.current.videoHeight || 400;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const base64 = canvas.toDataURL("image/jpeg", 0.9);
    setPreviewUrl(base64);
    stopWebcam();

    try {
      setLoading(true);
      setError(null);
      const res = await api.matchBase64(base64);
      setMatchResult(res);
    } catch (err: any) {
      setError(err.message || "Match failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    executeMatch(selectedFile);
  };

  const executeMatch = async (targetFile: File) => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.matchFile(targetFile);
      setMatchResult(res);
    } catch (err: any) {
      setError(err.message || "Match request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleAlphaChange = async (newAlpha: number) => {
    setAlpha(newAlpha);
    try {
      setAlphaUpdating(true);
      await api.setOperatingPoint(newAlpha);
      if (file) {
        await executeMatch(file);
      }
    } catch (err: any) {
      console.error(err);
    } finally {
      setAlphaUpdating(false);
    }
  };

  const runQuickTest = async (type: "genuine" | "lookalike") => {
    setError(null);
    try {
      setLoading(true);
      const filename =
        type === "genuine"
          ? "data/sample_market1501/query/0026_c4s1_002604_00.jpg"
          : "data/sample_market1501/bounding_box_test/0004_c3s1_000403_00.jpg";
      const sampleUrl = `${API_BASE}/static/${filename}`;
      const imgRes = await fetch(sampleUrl);
      const blob = await imgRes.blob();
      const testFile = new File([blob], `${type}_test.jpg`, { type: "image/jpeg" });
      setFile(testFile);
      setPreviewUrl(URL.createObjectURL(testFile));
      const res = await api.matchFile(testFile);
      setMatchResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to load sample probe");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-[1200px] mx-auto py-6 font-sans">
      {/* Top Banner: FAR Slider & One-Click Demo CTA */}
      <div className="border border-[#E5E7EB] rounded-lg p-5 bg-white flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-[#0A0A0A] tracking-tight">Open-Set Match & Verify</h1>
            <span className="text-[11px] font-mono bg-[#FAFAFA] border border-[#E5E7EB] px-2 py-0.5 rounded text-[#6B7280]">
              Dual-Barrier Engine
            </span>
          </div>
          <p className="text-xs text-[#6B7280] mt-1">
            Rejects unknown impostors and appearance look-alikes while accepting enrolled staff.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Target FAR (Alpha) Slider */}
          <div className="bg-[#FAFAFA] border border-[#E5E7EB] rounded-lg p-3 min-w-[270px]">
            <div className="flex justify-between items-center text-xs mb-1.5">
              <span className="font-semibold text-[#0A0A0A] flex items-center text-[11px]">
                <Sliders className="w-3.5 h-3.5 mr-1 text-[#6B7280]" />
                Target FAR (α):
              </span>
              <span className="font-bold font-mono tabular-nums text-[#0A0A0A] bg-white border border-[#E5E7EB] px-2 py-0.5 rounded text-[11px]">
                {(alpha * 100).toFixed(2)}%
              </span>
            </div>
            <input
              type="range"
              min="0.001"
              max="0.05"
              step="0.001"
              value={alpha}
              onChange={(e) => handleAlphaChange(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-[#E5E7EB] rounded-lg appearance-none cursor-pointer accent-[#0A0A0A]"
            />
            <div className="flex justify-between text-[10px] text-[#6B7280] font-mono mt-1">
              <span>0.1% (Strict)</span>
              <span>1.0% (Standard)</span>
              <span>5.0% (Relaxed)</span>
            </div>
          </div>

          {/* One-Click Demo Scenario CTA Button */}
          <button
            onClick={() => setDemoModalOpen(true)}
            className="inline-flex items-center px-4 py-2.5 bg-[#0A0A0A] text-white text-xs font-semibold rounded-lg hover:bg-black transition-colors focus:outline-none focus:ring-2 focus:ring-[#0A0A0A] focus:ring-offset-2"
          >
            <Play className="w-3.5 h-3.5 mr-2 fill-current" />
            Run Demo Scenario
          </button>
        </div>
      </div>

      {/* Error alert */}
      {error && (
        <div className="p-3.5 rounded-lg border border-red-200 bg-red-50 text-red-800 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <XCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="font-bold text-sm">×</button>
        </div>
      )}

      {/* Main Grid: Input Column (5 cols) & Result Column (7 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Probe Input */}
        <div className="lg:col-span-5 space-y-4">
          <div className="border border-[#E5E7EB] rounded-lg p-5 bg-white space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider">
                Probe Image Input
              </h2>
              <div className="flex space-x-1">
                <button
                  type="button"
                  onClick={() => {
                    stopWebcam();
                    fileInputRef.current?.click();
                  }}
                  className="px-2.5 py-1 text-xs border border-[#E5E7EB] rounded hover:bg-[#FAFAFA] font-medium text-[#0A0A0A]"
                >
                  Upload File
                </button>
                <button
                  type="button"
                  onClick={useWebcam ? stopWebcam : startWebcam}
                  className={`px-2.5 py-1 text-xs border border-[#E5E7EB] rounded font-medium ${
                    useWebcam ? "bg-[#0A0A0A] text-white" : "hover:bg-[#FAFAFA] text-[#0A0A0A]"
                  }`}
                >
                  <Camera className="w-3.5 h-3.5 inline mr-1" />
                  {useWebcam ? "Cancel" : "Webcam"}
                </button>
              </div>
            </div>

            {/* Webcam Live Feed */}
            {useWebcam ? (
              <div className="space-y-3">
                <div className="border border-[#E5E7EB] rounded-lg overflow-hidden bg-black flex items-center justify-center">
                  <video ref={videoRef} autoPlay playsInline className="w-full h-64 object-cover" />
                </div>
                <button
                  onClick={captureWebcamFrame}
                  className="w-full py-2 bg-[#0A0A0A] text-white text-xs font-semibold rounded-lg hover:bg-black transition-colors"
                >
                  Capture & Verify Frame
                </button>
              </div>
            ) : (
              /* Dropzone or Preview */
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files?.[0]) handleFileSelect(e.dataTransfer.files[0]);
                }}
                className="border-2 border-dashed border-[#E5E7EB] rounded-lg p-6 text-center cursor-pointer hover:border-[#0A0A0A] transition-colors bg-[#FAFAFA]"
              >
                {previewUrl ? (
                  <div className="space-y-2">
                    <img
                      src={previewUrl}
                      alt="Probe preview"
                      className="w-28 h-44 object-cover rounded-md mx-auto border border-[#E5E7EB] bg-white"
                    />
                    <p className="text-xs text-[#0A0A0A] font-medium">Click to select different crop</p>
                  </div>
                ) : (
                  <div>
                    <UploadCloud className="w-8 h-8 text-[#6B7280] mx-auto mb-2" />
                    <p className="text-xs font-medium text-[#0A0A0A]">
                      Click to upload probe crop or drag image here
                    </p>
                    <p className="text-[11px] text-[#6B7280] mt-1">256×128 Re-ID standard crop recommended</p>
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
                  }}
                  className="hidden"
                />
              </div>
            )}

            {/* Quick 1-Click Verification Test Buttons */}
            <div className="pt-4 border-t border-[#E5E7EB]">
              <span className="text-[11px] font-semibold text-[#6B7280] uppercase tracking-wider block mb-2">
                Quick Verification Probes
              </span>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => runQuickTest("genuine")}
                  className="p-2.5 border border-[#E5E7EB] rounded-lg bg-[#FAFAFA] hover:bg-white text-left text-xs transition-colors"
                >
                  <span className="font-semibold text-[#16A34A] block">✔ Genuine Staff</span>
                  <span className="text-[10px] text-[#6B7280]">Identity 24 (Expected: ACCEPTED)</span>
                </button>
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => runQuickTest("lookalike")}
                  className="p-2.5 border border-[#E5E7EB] rounded-lg bg-[#FAFAFA] hover:bg-white text-left text-xs transition-colors"
                >
                  <span className="font-semibold text-[#D97706] block">⚠ Look-Alike Impostor</span>
                  <span className="text-[10px] text-[#6B7280]">Appearance Twin (Expected: UNKNOWN)</span>
                </button>
              </div>
            </div>

            {/* Compare Toggle Control */}
            <div className="pt-4 border-t border-[#E5E7EB] flex items-center justify-between">
              <span className="text-xs text-[#0A0A0A] font-medium">
                Compare Plain Cosine vs Discern:
              </span>
              <button
                type="button"
                onClick={() => setShowComparison(!showComparison)}
                className="flex items-center text-xs text-[#0A0A0A] font-semibold focus:outline-none"
              >
                {showComparison ? (
                  <span className="text-[#16A34A] flex items-center">
                    <ToggleRight className="w-5 h-5 mr-1" />
                    Active
                  </span>
                ) : (
                  <span className="text-[#6B7280] flex items-center">
                    <ToggleLeft className="w-5 h-5 mr-1" />
                    Disabled
                  </span>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Decision & Verification Card */}
        <div className="lg:col-span-7 space-y-4">
          {loading ? (
            /* Skeleton Loader */
            <div className="border border-[#E5E7EB] rounded-lg p-8 bg-white space-y-4 animate-pulse">
              <div className="h-20 bg-[#FAFAFA] rounded-md border border-[#E5E7EB]" />
              <div className="grid grid-cols-4 gap-3">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="h-16 bg-[#FAFAFA] rounded-md border border-[#E5E7EB]" />
                ))}
              </div>
              <div className="h-24 bg-[#FAFAFA] rounded-md border border-[#E5E7EB]" />
              <div className="h-32 bg-[#FAFAFA] rounded-md border border-[#E5E7EB]" />
            </div>
          ) : !matchResult ? (
            /* Empty State */
            <div className="border border-[#E5E7EB] rounded-lg p-12 bg-white text-center space-y-3">
              <ShieldCheck className="w-10 h-10 text-[#6B7280] mx-auto" />
              <h3 className="text-sm font-semibold text-[#0A0A0A]">No Probe Evaluated Yet</h3>
              <p className="text-xs text-[#6B7280] max-w-sm mx-auto">
                Upload a person crop or select a quick test probe on the left to observe Discern's dual-barrier decision boundary.
              </p>
              <div className="pt-2">
                <button
                  onClick={() => runQuickTest("lookalike")}
                  className="px-3.5 py-1.5 bg-[#FAFAFA] border border-[#E5E7EB] text-xs font-semibold rounded-lg hover:bg-[#F3F4F6] text-[#0A0A0A]"
                >
                  Test Look-Alike Probe Now
                </button>
              </div>
            </div>
          ) : (
            /* Live Result Display */
            <div className="border border-[#E5E7EB] rounded-lg p-6 bg-white space-y-6">
              {/* Verdict Card */}
              <div
                className={`p-5 rounded-lg border ${
                  matchResult.decision === "ACCEPTED"
                    ? "bg-green-50/50 border-green-200"
                    : matchResult.margin < matchResult.operating_point.margin_delta
                    ? "bg-amber-50/50 border-amber-200"
                    : "bg-[#FAFAFA] border-[#E5E7EB]"
                }`}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center space-x-3">
                    {matchResult.decision === "ACCEPTED" ? (
                      <CheckCircle2 className="w-8 h-8 text-[#16A34A] flex-shrink-0" />
                    ) : matchResult.margin < matchResult.operating_point.margin_delta ? (
                      <ShieldAlert className="w-8 h-8 text-[#D97706] flex-shrink-0" />
                    ) : (
                      <XCircle className="w-8 h-8 text-[#DC2626] flex-shrink-0" />
                    )}
                    <div>
                      <div className="text-[10px] uppercase font-semibold tracking-wider text-[#6B7280]">
                        Identification Verdict
                      </div>
                      <div className="text-2xl font-bold text-[#0A0A0A] tracking-tight">
                        {matchResult.decision === "ACCEPTED"
                          ? `ACCEPTED as ${matchResult.predicted_name}`
                          : "UNKNOWN (REJECTED)"}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4 text-right">
                    <div>
                      <span className="text-[10px] uppercase font-semibold text-[#6B7280] block">
                        Confidence
                      </span>
                      <span className="text-2xl font-bold font-mono tabular-nums text-[#0A0A0A]">
                        {(matchResult.calibrated_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    {matchResult.inference_time_ms != null && (
                      <div className="text-right border-l border-[#E5E7EB] pl-3">
                        <span className="text-[10px] uppercase font-semibold text-[#6B7280] block flex items-center justify-end">
                          <Clock className="w-3 h-3 mr-0.5" /> Latency
                        </span>
                        <span className="text-sm font-mono tabular-nums text-[#0A0A0A]">
                          {matchResult.inference_time_ms} ms
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Plain-English Reason Box */}
                <div className="mt-4 pt-3 border-t border-black/5 text-xs text-[#0A0A0A] leading-relaxed font-medium">
                  <strong>Decision Reason:</strong> {matchResult.human_reason}
                </div>
              </div>

              {/* Side-by-Side Images: Query vs Matched Gallery Candidates */}
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider block">
                  Probe Image vs Enrolled Gallery Candidates
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center border border-[#E5E7EB] rounded-lg p-3 bg-[#FAFAFA]">
                  {/* Query Image */}
                  <div className="sm:col-span-4 text-center">
                    <span className="text-[10px] font-semibold text-[#6B7280] uppercase block mb-1">
                      Query Probe
                    </span>
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt="Query probe"
                        className="w-20 h-36 object-cover rounded mx-auto border border-[#E5E7EB] bg-white"
                      />
                    ) : (
                      <div className="w-20 h-36 bg-white border border-[#E5E7EB] rounded mx-auto flex items-center justify-center text-[10px] text-[#6B7280]">
                        Probe
                      </div>
                    )}
                  </div>

                  {/* Arrow Indicator */}
                  <div className="sm:col-span-1 flex items-center justify-center text-[#6B7280]">
                    <ArrowRight className="w-4 h-4 hidden sm:block" />
                  </div>

                  {/* Matched Gallery Candidates */}
                  <div className="sm:col-span-7 space-y-2">
                    <span className="text-[10px] font-semibold text-[#6B7280] uppercase block">
                      Top Gallery Matches
                    </span>
                    <div className="grid grid-cols-2 gap-2">
                      {matchResult.top_candidates.slice(0, 2).map((cand) => (
                        <div
                          key={cand.identity_id}
                          className="p-2 border border-[#E5E7EB] rounded bg-white flex items-center space-x-2"
                        >
                          {cand.thumbnail_url ? (
                            <img
                              src={`${API_BASE}${cand.thumbnail_url}`}
                              alt={cand.name}
                              className="w-10 h-16 object-cover rounded border border-[#E5E7EB]"
                            />
                          ) : (
                            <div className="w-10 h-16 bg-[#FAFAFA] border border-[#E5E7EB] rounded flex items-center justify-center text-[9px] text-[#6B7280]">
                              #{cand.rank}
                            </div>
                          )}
                          <div className="flex-1 min-w-0">
                            <span className="text-xs font-semibold text-[#0A0A0A] block truncate">
                              {cand.name}
                            </span>
                            <span className="text-[10px] text-[#6B7280] font-mono block">
                              Rank {cand.rank}
                            </span>
                            <span className="text-xs font-mono tabular-nums font-bold text-[#0A0A0A] mt-0.5 block">
                              {(cand.similarity * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Horizontal Similarity Bar */}
              <div className="border border-[#E5E7EB] rounded-lg p-4 bg-white">
                <HorizontalSimilarityBar
                  top1Similarity={matchResult.raw_similarity}
                  top2Similarity={matchResult.competitor_similarity}
                  thresholdTau={matchResult.operating_point.threshold_tau}
                  marginDelta={matchResult.operating_point.margin_delta}
                  decision={matchResult.decision}
                  top1Name={matchResult.top_candidates[0]?.name || "Top-1"}
                  top2Name={matchResult.top_candidates[1]?.name || "Top-2"}
                />
              </div>

              {/* Compare Toggle Box (Plain Cosine vs Discern) */}
              {showComparison && matchResult.baseline_comparison && (
                <div className="border border-[#E5E7EB] rounded-lg p-4 bg-[#FAFAFA] space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider">
                      Side-by-Side Comparison: Plain Cosine vs Discern
                    </span>
                    {matchResult.baseline_comparison.is_false_accept && (
                      <span className="text-[11px] font-semibold text-[#DC2626] bg-red-50 border border-red-200 px-2 py-0.5 rounded">
                        Baseline False Accept Avoided!
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                    {/* Baseline Cosine Card */}
                    <div
                      className={`p-3 border rounded-lg bg-white ${
                        matchResult.baseline_comparison.is_false_accept
                          ? "border-red-300 ring-1 ring-red-200"
                          : "border-[#E5E7EB]"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-semibold text-[#6B7280]">Plain Cosine Baseline</span>
                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                            matchResult.baseline_comparison.decision === "ACCEPTED"
                              ? matchResult.baseline_comparison.is_false_accept
                                ? "bg-red-50 text-[#DC2626] border border-red-200"
                                : "bg-green-50 text-[#16A34A] border border-green-200"
                              : "bg-[#FAFAFA] text-[#6B7280] border border-[#E5E7EB]"
                          }`}
                        >
                          {matchResult.baseline_comparison.decision}
                          {matchResult.baseline_comparison.is_false_accept && " (FALSE ACCEPT)"}
                        </span>
                      </div>
                      <p className="text-[11px] text-[#6B7280] leading-relaxed">
                        {matchResult.baseline_comparison.explanation}
                      </p>
                      <div className="mt-2 text-[10px] text-[#6B7280] font-mono">
                        Fixed Threshold: {(matchResult.baseline_comparison.threshold * 100).toFixed(1)}% • Competitor Margin: Ignored
                      </div>
                    </div>

                    {/* Discern Dual-Barrier Card */}
                    <div className="p-3 border border-green-300 rounded-lg bg-white ring-1 ring-green-100">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-semibold text-[#0A0A0A]">Discern (Dual-Barrier)</span>
                        <span
                          className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                            matchResult.decision === "ACCEPTED"
                              ? "bg-green-50 text-[#16A34A] border border-green-200"
                              : "bg-[#FAFAFA] text-[#0A0A0A] border border-[#E5E7EB]"
                          }`}
                        >
                          {matchResult.decision} (PROTECTED)
                        </span>
                      </div>
                      <p className="text-[11px] text-[#6B7280] leading-relaxed">
                        {matchResult.decision === "ACCEPTED"
                          ? `Passed calibrated similarity (${(matchResult.raw_similarity * 100).toFixed(1)}% >= ${(matchResult.operating_point.threshold_tau * 100).toFixed(1)}%) with strong safety margin (${(matchResult.margin * 100).toFixed(1)}% >= ${(matchResult.operating_point.margin_delta * 100).toFixed(1)}%).`
                          : `Refused admission: dual-barrier algorithm detects ambiguity between appearance twins (margin ${(matchResult.margin * 100).toFixed(1)}% < ${(matchResult.operating_point.margin_delta * 100).toFixed(1)}%).`}
                      </p>
                      <div className="mt-2 text-[10px] text-[#0A0A0A] font-mono">
                        Adaptive τ: {(matchResult.operating_point.threshold_tau * 100).toFixed(1)}% • Safety Margin δ: {(matchResult.operating_point.margin_delta * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Demo Scenario Modal */}
      <DemoScenarioModal isOpen={demoModalOpen} onClose={() => setDemoModalOpen(false)} />
    </div>
  );
};

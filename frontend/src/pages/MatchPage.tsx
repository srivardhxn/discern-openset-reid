import React, { useState, useRef, useEffect } from "react";
import {
  api,
  MatchResponse,
  IdentifyResponse,
  OperatingPoint,
  normaliseIdentify,
  API_BASE,
} from "../api";
import { HorizontalSimilarityBar } from "../components/HorizontalSimilarityBar";
import {
  UploadCloud,
  Camera,
  CheckCircle2,
  XCircle,
  Sliders,
  ArrowRight,
  ShieldCheck,
  ShieldAlert,
  Play,
  Clock,
  Users,
} from "lucide-react";

const OP_OPTIONS: { value: OperatingPoint; label: string; far: string; tar: string }[] = [
  { value: "strict",   label: "Strict",   far: "0.10%", tar: "8.8%"  },
  { value: "balanced", label: "Balanced", far: "0.99%", tar: "51.4%" },
  { value: "lenient",  label: "Lenient",  far: "5.00%", tar: "82.7%" },
];

export const MatchPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [rawResult, setRawResult] = useState<IdentifyResponse | null>(null);
  const [matchResult, setMatchResult] = useState<MatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Operating point (strict / balanced / lenient)
  const [op, setOp] = useState<OperatingPoint>("balanced");

  // Demo Modal
  const [demoModalOpen, setDemoModalOpen] = useState<boolean>(false);

  // Demo load state
  const [demoLoaded, setDemoLoaded] = useState<boolean>(false);
  const [demoLoading, setDemoLoading] = useState<boolean>(false);

  // Webcam state
  const [useWebcam, setUseWebcam] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Check if demo is already loaded
    api.getDemoStatus().then((s) => {
      if (s.enrolled_identities > 0) setDemoLoaded(true);
    }).catch(() => {});
    return () => { stopWebcam(); };
  }, []);

  const startWebcam = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 640 } });
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
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setUseWebcam(false);
  };

  const captureWebcamFrame = async () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width  = videoRef.current.videoWidth  || 300;
    canvas.height = videoRef.current.videoHeight || 400;
    canvas.getContext("2d")?.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const base64 = canvas.toDataURL("image/jpeg", 0.9);
    setPreviewUrl(base64);
    stopWebcam();
    try {
      setLoading(true); setError(null);
      const res = await api.matchBase64(base64);
      setMatchResult(res);
    } catch (err: any) {
      setError(err.message || "Match failed");
    } finally {
      setLoading(false);
    }
  };

  const executeMatch = async (targetFile: File, currentOp: OperatingPoint) => {
    try {
      setLoading(true); setError(null);
      const raw = await api.identifyFile(targetFile, currentOp);
      setRawResult(raw);
      setMatchResult(normaliseIdentify(raw));
    } catch (err: any) {
      setError(err.message || "Match request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    executeMatch(selectedFile, op);
  };

  const handleOpChange = async (newOp: OperatingPoint) => {
    setOp(newOp);
    if (file) await executeMatch(file, newOp);
  };

  const loadDemo = async () => {
    setDemoLoading(true);
    setError(null);
    try {
      const res = await api.loadDemo();
      setDemoLoaded(true);
      setError(null);
      alert(`Demo loaded: ${res.enrolled_identities} identities (${res.method})`);
    } catch (err: any) {
      setError(err.message || "Demo load failed");
    } finally {
      setDemoLoading(false);
    }
  };

  // Convenience: fetch a sample from static files and run identify
  const runQuickTest = async (type: "genuine" | "lookalike") => {
    setError(null);
    try {
      setLoading(true);
      // Use real probe from demo gallery
      const samplePath =
        type === "genuine"
          ? "demo/images/1265_c1s5_050816_00.jpg"   // Enrolled ID-1265 probe
          : "demo/images/0678_c1s3_063451_00.jpg";  // Unenrolled look-alike probe
      const sampleUrl = `${API_BASE}/static/${samplePath}`;
      const imgRes = await fetch(sampleUrl);
      const blob = await imgRes.blob();
      const testFile = new File([blob], `${type}_test.jpg`, { type: "image/jpeg" });
      setFile(testFile);
      setPreviewUrl(URL.createObjectURL(testFile));
      const raw = await api.identifyFile(testFile, op);
      setRawResult(raw);
      setMatchResult(normaliseIdentify(raw));
    } catch (err: any) {
      setError(err.message || "Failed to load sample probe");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-[1200px] mx-auto py-6 font-sans">

      {/* ── Top Banner ───────────────────────────────────────────────────── */}
      <div className="border border-[#E5E7EB] rounded-lg p-5 bg-white flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-[#0A0A0A] tracking-tight">Open-Set Match & Verify</h1>
            <span className="text-[11px] font-mono bg-[#FAFAFA] border border-[#E5E7EB] px-2 py-0.5 rounded text-[#6B7280]">
              ONNX · Calibrated · Flip-TTA
            </span>
          </div>
          <p className="text-xs text-[#6B7280] mt-1">
            Rejects unknown impostors and look-alikes using calibrated confidence + operating-point threshold.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Operating-Point Selector */}
          <div className="bg-[#FAFAFA] border border-[#E5E7EB] rounded-lg p-3 min-w-[300px]">
            <div className="flex items-center space-x-1.5 mb-2">
              <Sliders className="w-3.5 h-3.5 text-[#6B7280]" />
              <span className="text-[11px] font-semibold text-[#0A0A0A]">Operating Point</span>
            </div>
            <div className="flex rounded-md overflow-hidden border border-[#E5E7EB]">
              {OP_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  onClick={() => handleOpChange(o.value)}
                  className={`flex-1 py-1.5 text-[11px] font-semibold transition-colors ${
                    op === o.value
                      ? "bg-[#0A0A0A] text-white"
                      : "bg-white text-[#6B7280] hover:bg-[#FAFAFA]"
                  }`}
                >
                  {o.label}
                </button>
              ))}
            </div>
            {(() => {
              const cur = OP_OPTIONS.find((o) => o.value === op)!;
              return (
                <div className="flex justify-between mt-1.5 text-[10px] text-[#6B7280] font-mono">
                  <span>FAR ≤ {cur.far}</span>
                  <span>TAR ≈ {cur.tar}</span>
                  <span>
                    τ = {rawResult?.threshold?.toFixed(3) ?? "…"}
                  </span>
                </div>
              );
            })()}
          </div>

          {/* Demo loader */}
          {!demoLoaded ? (
            <button
              onClick={loadDemo}
              disabled={demoLoading}
              className="inline-flex items-center px-4 py-2.5 bg-[#0A0A0A] text-white text-xs font-semibold rounded-lg hover:bg-black transition-colors disabled:opacity-60"
            >
              <Play className="w-3.5 h-3.5 mr-2 fill-current" />
              {demoLoading ? "Loading Demo…" : "Load Demo Gallery"}
            </button>
          ) : (
            <span className="text-xs font-semibold text-[#16A34A] flex items-center bg-green-50 border border-green-200 px-3 py-2 rounded-lg">
              <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A] mr-1.5" />
              Demo Gallery Loaded
            </span>
          )}
        </div>
      </div>

      {/* ── Error alert ──────────────────────────────────────────────────── */}
      {error && (
        <div className="p-3.5 rounded-lg border border-red-200 bg-red-50 text-red-800 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <XCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="font-bold text-sm">×</button>
        </div>
      )}

      {/* ── Main Grid ────────────────────────────────────────────────────── */}
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
                  onClick={() => { stopWebcam(); fileInputRef.current?.click(); }}
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

            {/* Webcam feed */}
            {useWebcam ? (
              <div className="space-y-3">
                <div className="border border-[#E5E7EB] rounded-lg overflow-hidden bg-black">
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
                onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files?.[0]) handleFileSelect(e.dataTransfer.files[0]); }}
                className="border-2 border-dashed border-[#E5E7EB] rounded-lg p-6 text-center cursor-pointer hover:border-[#0A0A0A] transition-colors bg-[#FAFAFA]"
              >
                {previewUrl ? (
                  <div className="space-y-2">
                    <img src={previewUrl} alt="Probe" className="w-28 h-44 object-cover rounded-md mx-auto border border-[#E5E7EB] bg-white" />
                    <p className="text-xs text-[#0A0A0A] font-medium">Click to select different crop</p>
                  </div>
                ) : (
                  <div>
                    <UploadCloud className="w-8 h-8 text-[#6B7280] mx-auto mb-2" />
                    <p className="text-xs font-medium text-[#0A0A0A]">Click to upload probe crop or drag image here</p>
                    <p className="text-[11px] text-[#6B7280] mt-1">144×288 Re-ID person crop recommended</p>
                  </div>
                )}
                <input ref={fileInputRef} type="file" accept="image/*" onChange={(e) => { if (e.target.files?.[0]) handleFileSelect(e.target.files[0]); }} className="hidden" />
              </div>
            )}

            {/* Quick test buttons */}
            <div className="pt-4 border-t border-[#E5E7EB]">
              <span className="text-[11px] font-semibold text-[#6B7280] uppercase tracking-wider block mb-2">
                Quick Verification Probes (from samples/)
              </span>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => runQuickTest("genuine")}
                  className="p-2.5 border border-[#E5E7EB] rounded-lg bg-[#FAFAFA] hover:bg-white text-left text-xs transition-colors"
                >
                  <span className="font-semibold text-[#16A34A] block">✔ Genuine Pair</span>
                  <span className="text-[10px] text-[#6B7280]">ID-1265 enrolled subject (Expected: ACCEPTED)</span>
                </button>
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => runQuickTest("lookalike")}
                  className="p-2.5 border border-[#E5E7EB] rounded-lg bg-[#FAFAFA] hover:bg-white text-left text-xs transition-colors"
                >
                  <span className="font-semibold text-[#D97706] block">⚠ Look-Alike Impostor</span>
                  <span className="text-[10px] text-[#6B7280]">ID-0678 stranger look-alike (Expected: UNKNOWN)</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Decision & Scores */}
        <div className="lg:col-span-7 space-y-4">
          {loading ? (
            <div className="border border-[#E5E7EB] rounded-lg p-8 bg-white space-y-4 animate-pulse">
              <div className="h-20 bg-[#FAFAFA] rounded-md border border-[#E5E7EB]" />
              <div className="grid grid-cols-4 gap-3">
                {[1, 2, 3, 4].map((i) => <div key={i} className="h-16 bg-[#FAFAFA] rounded-md border border-[#E5E7EB]" />)}
              </div>
              <div className="h-24 bg-[#FAFAFA] rounded-md border border-[#E5E7EB]" />
            </div>
          ) : !matchResult ? (
            <div className="border border-[#E5E7EB] rounded-lg p-12 bg-white text-center space-y-3">
              <ShieldCheck className="w-10 h-10 text-[#6B7280] mx-auto" />
              <h3 className="text-sm font-semibold text-[#0A0A0A]">No Probe Evaluated Yet</h3>
              <p className="text-xs text-[#6B7280] max-w-sm mx-auto">
                Upload a person crop or load the demo gallery and select a quick test probe.
              </p>
              {!demoLoaded && (
                <button
                  onClick={loadDemo}
                  disabled={demoLoading}
                  className="px-3.5 py-1.5 bg-[#0A0A0A] text-white border border-[#E5E7EB] text-xs font-semibold rounded-lg hover:bg-black"
                >
                  {demoLoading ? "Loading…" : "Load Demo Gallery First"}
                </button>
              )}
            </div>
          ) : (
            <div className="border border-[#E5E7EB] rounded-lg p-6 bg-white space-y-6">

              {/* Verdict Card */}
              <div className={`p-5 rounded-lg border ${
                matchResult.decision === "ACCEPTED"
                  ? "bg-green-50/50 border-green-200"
                  : "bg-[#FAFAFA] border-[#E5E7EB]"
              }`}>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center space-x-3">
                    {matchResult.decision === "ACCEPTED"
                      ? <CheckCircle2 className="w-8 h-8 text-[#16A34A] flex-shrink-0" />
                      : <XCircle className="w-8 h-8 text-[#DC2626] flex-shrink-0" />
                    }
                    <div>
                      <div className="text-[10px] uppercase font-semibold tracking-wider text-[#6B7280]">
                        Identification Verdict
                      </div>
                      <div className="text-2xl font-bold text-[#0A0A0A] tracking-tight">
                        {matchResult.decision === "ACCEPTED"
                          ? `ACCEPTED — ${matchResult.predicted_name}`
                          : "UNKNOWN"}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4 text-right">
                    <div>
                      <span className="text-[10px] uppercase font-semibold text-[#6B7280] block">Confidence</span>
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

                {/* s1 / margin / z scores */}
                {rawResult && (
                  <div className="mt-4 pt-3 border-t border-black/5 grid grid-cols-3 gap-3">
                    {[
                      { label: "s₁ (best identity score)", value: rawResult.s1.toFixed(4) },
                      { label: "margin (s₁ − s₂)", value: rawResult.margin.toFixed(4) },
                      { label: "z-score", value: rawResult.z.toFixed(3) },
                    ].map((item) => (
                      <div key={item.label} className="bg-white border border-[#E5E7EB] rounded-md p-2.5 text-center">
                        <span className="text-[10px] text-[#6B7280] block leading-tight">{item.label}</span>
                        <span className="text-sm font-bold font-mono text-[#0A0A0A]">{item.value}</span>
                      </div>
                    ))}
                  </div>
                )}

                {rawResult && (
                  <div className="mt-2 text-[10px] text-[#6B7280] font-mono">
                    Calibrator: {rawResult.calibrator} · τ = {rawResult.threshold.toFixed(4)} · op = {rawResult.operating_point}
                  </div>
                )}
              </div>

              {/* Top-5 candidates */}
              {rawResult?.top5 && rawResult.top5.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider block">
                    Top-5 Candidates
                  </span>
                  <div className="border border-[#E5E7EB] rounded-lg overflow-hidden">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-[#FAFAFA] border-b border-[#E5E7EB]">
                          <th className="text-left px-3 py-2 font-semibold text-[#6B7280]">Rank</th>
                          <th className="text-left px-3 py-2 font-semibold text-[#6B7280]">Identity</th>
                          <th className="text-right px-3 py-2 font-semibold text-[#6B7280]">Score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#E5E7EB]">
                        {rawResult.top5.map((t, i) => (
                          <tr key={i} className={i === 0 && matchResult.decision === "ACCEPTED" ? "bg-green-50/40" : ""}>
                            <td className="px-3 py-2 font-mono text-[#6B7280]">#{i + 1}</td>
                            <td className="px-3 py-2 font-medium text-[#0A0A0A]">
                              {t.identity}
                              {i === 0 && matchResult.decision === "ACCEPTED" && (
                                <span className="ml-1.5 text-[10px] text-[#16A34A] font-semibold bg-green-100 px-1.5 py-0.5 rounded">MATCH</span>
                              )}
                            </td>
                            <td className="px-3 py-2 text-right font-mono font-bold text-[#0A0A0A]">
                              {(t.score * 100).toFixed(2)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Probe vs gallery images side-by-side */}
              {matchResult.top_candidates.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider block">
                    Probe vs Top Candidates
                  </span>
                  <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center border border-[#E5E7EB] rounded-lg p-3 bg-[#FAFAFA]">
                    <div className="sm:col-span-3 text-center">
                      <span className="text-[10px] font-semibold text-[#6B7280] uppercase block mb-1">Query</span>
                      {previewUrl ? (
                        <img src={previewUrl} alt="Query" className="w-20 h-36 object-cover rounded mx-auto border border-[#E5E7EB] bg-white" />
                      ) : (
                        <div className="w-20 h-36 bg-white border border-[#E5E7EB] rounded mx-auto flex items-center justify-center text-[10px] text-[#6B7280]">Probe</div>
                      )}
                    </div>
                    <div className="sm:col-span-1 flex items-center justify-center text-[#6B7280]">
                      <ArrowRight className="w-4 h-4 hidden sm:block" />
                    </div>
                    <div className="sm:col-span-8 space-y-2">
                      <span className="text-[10px] font-semibold text-[#6B7280] uppercase block">Gallery Candidates</span>
                      <div className="grid grid-cols-2 gap-2">
                        {matchResult.top_candidates.slice(0, 2).map((cand) => (
                          <div key={cand.identity_id} className="p-2 border border-[#E5E7EB] rounded bg-white flex items-center space-x-2">
                            <div className="w-10 h-16 bg-[#FAFAFA] border border-[#E5E7EB] rounded flex items-center justify-center text-[9px] text-[#6B7280] font-mono">
                              #{cand.rank}
                            </div>
                            <div className="flex-1 min-w-0">
                              <span className="text-xs font-semibold text-[#0A0A0A] block truncate">{cand.name}</span>
                              <span className="text-[10px] text-[#6B7280] font-mono block">Rank {cand.rank}</span>
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
              )}

              {/* Similarity bar */}
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

            </div>
          )}
        </div>
      </div>
    </div>
  );
};

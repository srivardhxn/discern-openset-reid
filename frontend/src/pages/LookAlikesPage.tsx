import React, { useEffect, useState } from "react";
import { api, LookAlikePair, LookalikesResponse, API_BASE } from "../api";
import { Eye, ShieldAlert, AlertCircle, CheckCircle2 } from "lucide-react";

export const LookAlikesPage: React.FC = () => {
  const [data, setData] = useState<LookalikesResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { loadLookAlikes(); }, []);

  const loadLookAlikes = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getLookalikes();
      setData(res);
    } catch (err: any) {
      setError(err.message || "Failed to load look-alike pairs");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-[1200px] mx-auto py-12 text-center font-sans">
        <div className="w-8 h-8 border-2 border-[#0A0A0A] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-xs text-[#6B7280]">Loading lookalike_explorer.json…</p>
      </div>
    );
  }

  if (error || !data || !data.pairs?.length) {
    return (
      <div className="max-w-[1200px] mx-auto py-12 font-sans">
        <div className="p-8 border border-[#E5E7EB] rounded-lg text-center bg-[#FAFAFA]">
          <AlertCircle className="w-8 h-8 text-[#D97706] mx-auto mb-2" />
          <h2 className="text-sm font-semibold text-[#0A0A0A]">No Look-Alike Data Available</h2>
          <p className="text-xs text-[#6B7280] mt-1">
            {error || "lookalike_explorer.json is empty or not found in the model bundle."}
          </p>
        </div>
      </div>
    );
  }

  const pairs = data.pairs;
  const ops   = data.operating_points ?? {};

  return (
    <div className="space-y-6 max-w-[1200px] mx-auto py-6 font-sans">

      {/* ── Header ───────────────────────────────────────────────────────── */}
      <div className="border border-[#E5E7EB] rounded-lg p-5 bg-white">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center space-x-2">
              <Eye className="w-4 h-4 text-[#0A0A0A]" />
              <h1 className="text-xl font-bold text-[#0A0A0A] tracking-tight">
                Look-Alike Explorer
              </h1>
              <span className="text-[11px] font-mono bg-[#FAFAFA] border border-[#E5E7EB] px-2 py-0.5 rounded text-[#6B7280]">
                lookalike_explorer.json
              </span>
            </div>
            <p className="text-xs text-[#6B7280] mt-1 max-w-3xl">
              {data.description}
            </p>
          </div>
          <div className="flex items-center space-x-3 text-xs font-mono tabular-nums text-[#6B7280]">
            <span className="px-2.5 py-1 bg-[#FAFAFA] border border-[#E5E7EB] rounded">
              <strong className="text-[#0A0A0A]">{pairs.length}</strong> Pairs
            </span>
          </div>
        </div>

        {/* Operating-point thresholds from the explorer */}
        {Object.keys(ops).length > 0 && (
          <div className="mt-4 pt-4 border-t border-[#E5E7EB] grid grid-cols-3 gap-3">
            {Object.entries(ops).map(([opName, opData]: [string, any]) => (
              <div key={opName} className="bg-[#FAFAFA] border border-[#E5E7EB] rounded-lg p-3">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-[#0A0A0A] mb-1.5 capitalize">
                  {opName}
                </div>
                <div className="space-y-0.5 font-mono text-[10px] text-[#6B7280]">
                  <div>FAR: {(opData.far * 100).toFixed(2)}%</div>
                  <div>FAR look-alike: {(opData.far_lookalike_lv90 * 100).toFixed(2)}%</div>
                  <div>TAR: {(opData.tar * 100).toFixed(1)}%</div>
                  <div>τ = {opData.threshold.toFixed(4)}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Pair Cards ───────────────────────────────────────────────────── */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider">
            Most Confusable Different-Identity Pairs
          </h2>
          <span className="text-xs text-[#6B7280]">Ranked by identity-centroid cosine similarity</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {pairs.map((pair) => (
            <LookAlikePairCard key={pair.rank} pair={pair} />
          ))}
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Single pair card
// ---------------------------------------------------------------------------
function LookAlikePairCard({ pair }: { pair: LookAlikePair }) {
  const imgA = pair.image_a_url;
  const imgB = pair.image_b_url;
  const genA = pair.genuine_image_1_url;
  const genB = pair.genuine_image_2_url;

  return (
    <div className="border border-[#E5E7EB] rounded-lg p-4 bg-white space-y-4">
      {/* Card header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-bold text-[#0A0A0A]">
          Rank #{pair.rank}: ID-{pair.id_a} vs ID-{pair.id_b}
        </span>
        <span className="text-xs font-mono font-bold text-[#D97706] bg-amber-50 border border-amber-200 px-2 py-0.5 rounded">
          Centroid sim: {(pair.identity_centroid_similarity * 100).toFixed(1)}%
        </span>
      </div>

      {/* ── Impostor pair (SHOULD be rejected) ───────────────────────── */}
      <div>
        <div className="flex items-center space-x-1.5 mb-2">
          <ShieldAlert className="w-3.5 h-3.5 text-[#D97706]" />
          <span className="text-[11px] font-semibold text-[#0A0A0A]">Impostor Pair (must → UNKNOWN)</span>
          <span className="text-[10px] text-[#6B7280] font-mono">
            sim = {pair.impostor_similarity.toFixed(3)} · conf = {(pair.impostor_confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3 border border-amber-200 bg-amber-50/30 rounded-lg p-3">
          <PersonThumb label={`ID-${pair.id_a}`} url={imgA} />
          <PersonThumb label={`ID-${pair.id_b}`} url={imgB} />
        </div>
        <div className="mt-2 text-[10px] text-[#6B7280]">
          Colour similarity: {(pair.colour_similarity * 100).toFixed(1)}% · Both flagged as DIFFERENT identities.
          Discern must reject; plain cosine often false-accepts at this confidence.
        </div>
      </div>

      {/* ── Genuine pair (SHOULD be accepted) ───────────────────────── */}
      {genA && genB && (
        <div>
          <div className="flex items-center space-x-1.5 mb-2">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#16A34A]" />
            <span className="text-[11px] font-semibold text-[#0A0A0A]">
              Genuine Pair of ID-{pair.id_a} (must → ACCEPTED)
            </span>
            <span className="text-[10px] text-[#6B7280] font-mono">
              sim = {pair.genuine_similarity.toFixed(3)} · conf = {(pair.genuine_confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 border border-green-200 bg-green-50/30 rounded-lg p-3">
            <PersonThumb label="Camera A" url={genA} />
            <PersonThumb label="Camera B" url={genB} />
          </div>
          <div className="mt-2 text-[10px] text-[#6B7280]">
            Cross-camera genuine match: same person, different cameras. Confidence is lower than the impostor
            pair — illustrating the look-alike challenge (colour twins are hard to distinguish).
          </div>
        </div>
      )}
    </div>
  );
}

function PersonThumb({ label, url }: { label: string; url?: string }) {
  return (
    <div className="text-center">
      <span className="text-[11px] font-semibold text-[#0A0A0A] block mb-1">{label}</span>
      {url ? (
        <img
          src={`${API_BASE}${url}`}
          alt={label}
          className="w-20 h-36 object-cover rounded mx-auto border border-[#E5E7EB] bg-white"
          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
        />
      ) : (
        <div className="w-20 h-36 bg-white border border-[#E5E7EB] rounded mx-auto flex items-center justify-center text-[10px] text-[#6B7280]">
          No Image
        </div>
      )}
    </div>
  );
}

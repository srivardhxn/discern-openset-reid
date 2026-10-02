import React, { useState, useEffect, useRef } from "react";
import { api, GalleryIdentity, API_BASE } from "../api";
import { UploadCloud, Trash2, Users, AlertCircle, CheckCircle, RefreshCw, Eye } from "lucide-react";

export const EnrollPage: React.FC<{ onGalleryChange?: () => void }> = ({ onGalleryChange }) => {
  const [gallery, setGallery] = useState<GalleryIdentity[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [name, setName] = useState<string>("");
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState<boolean>(false);
  const [msg, setMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<{ id: number; name: string } | null>(null);
  const [deleting, setDeleting] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadGallery();
  }, []);

  const loadGallery = async () => {
    try {
      setLoading(true);
      const res = await api.getGallery();
      setGallery(res);
      if (onGalleryChange) onGalleryChange();
    } catch (err: any) {
      setMsg({ type: "error", text: err.message || "Failed to load gallery" });
    } finally {
      setLoading(false);
    }
  };

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const selected = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
      setFiles((prev) => [...prev, ...selected]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selected = Array.from(e.target.files);
      setFiles((prev) => [...prev, ...selected]);
    }
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setMsg({ type: "error", text: "Please enter an identity name" });
      return;
    }
    if (files.length === 0) {
      setMsg({ type: "error", text: "Please select at least one image" });
      return;
    }

    try {
      setUploading(true);
      setMsg(null);
      await api.enrollIdentity(name.trim(), files);
      setName("");
      setFiles([]);
      setMsg({ type: "success", text: `Successfully enrolled identity '${name}'. Gallery whitening and prototypes updated.` });
      await loadGallery();
    } catch (err: any) {
      setMsg({ type: "error", text: err.message || "Enrollment failed" });
    } finally {
      setUploading(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    try {
      setDeleting(true);
      await api.deleteIdentity(deleteTarget.id);
      setMsg({ type: "success", text: `Identity '${deleteTarget.name}' (ID ${deleteTarget.id}) deleted. Whitening re-fitted.` });
      setDeleteTarget(null);
      await loadGallery();
    } catch (err: any) {
      setMsg({ type: "error", text: err.message || "Failed to delete identity" });
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-[1200px] mx-auto py-6 font-sans">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#E5E7EB] gap-2">
        <div>
          <h1 className="text-xl font-bold text-[#0A0A0A] tracking-tight">Gallery & Identity Enrollment</h1>
          <p className="text-xs text-[#6B7280] mt-0.5">
            Enrolling or removing identities automatically re-computes whitening matrices and per-identity adaptive thresholds.
          </p>
        </div>
        <button
          onClick={loadGallery}
          className="inline-flex items-center px-3 py-1.5 text-xs font-medium text-[#0A0A0A] bg-[#FAFAFA] border border-[#E5E7EB] rounded hover:bg-[#F3F4F6] transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${loading ? "animate-spin" : ""}`} />
          Refresh Gallery
        </button>
      </div>

      {/* Notifications */}
      {msg && (
        <div
          className={`p-4 rounded border text-xs flex items-center justify-between ${
            msg.type === "success"
              ? "bg-green-50 border-green-200 text-green-800"
              : "bg-red-50 border-red-200 text-red-800"
          }`}
        >
          <div className="flex items-center space-x-2">
            {msg.type === "success" ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
            <span>{msg.text}</span>
          </div>
          <button onClick={() => setMsg(null)} className="font-semibold ml-2">×</button>
        </div>
      )}

      {/* Enrollment Card */}
      <div className="border border-[#E5E7EB] rounded p-6 bg-white">
        <h2 className="text-sm font-semibold text-[#111827] uppercase tracking-wider mb-4">
          Enroll New Identity
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-[#111827] mb-1">
              Identity Name / Label <span className="text-[#DC2626]">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Officer John Davis (Badge #401)"
              className="w-full px-3 py-2 text-sm border border-[#E5E7EB] rounded focus:outline-none focus:border-[#111827] bg-white"
            />
          </div>

          {/* Drag & Drop Area */}
          <div>
            <label className="block text-xs font-medium text-[#111827] mb-1">
              Person Images (Crops) <span className="text-[#DC2626]">*</span>
            </label>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-[#E5E7EB] rounded p-6 text-center cursor-pointer hover:border-[#111827] transition-colors bg-[#F9FAFB]"
            >
              <UploadCloud className="w-8 h-8 text-[#6B7280] mx-auto mb-2" />
              <p className="text-xs font-medium text-[#111827]">
                Click to browse or drag and drop images here
              </p>
              <p className="text-[11px] text-[#6B7280] mt-1">Supports JPEG, PNG. Multiple images improve exemplar coverage.</p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          </div>

          {/* Staged file previews */}
          {files.length > 0 && (
            <div>
              <div className="text-xs font-medium text-[#111827] mb-2">
                Selected Images ({files.length}):
              </div>
              <div className="flex flex-wrap gap-2">
                {files.map((f, idx) => (
                  <div key={idx} className="relative group border border-[#E5E7EB] rounded p-1 bg-white">
                    <img
                      src={URL.createObjectURL(f)}
                      alt="preview"
                      className="w-16 h-24 object-cover rounded"
                    />
                    <button
                      type="button"
                      onClick={() => removeFile(idx)}
                      className="absolute -top-1.5 -right-1.5 bg-[#DC2626] text-white rounded-full w-4 h-4 flex items-center justify-center text-[10px]"
                    >
                      ×
                    </button>
                    <span className="block text-[9px] text-[#6B7280] truncate max-w-[64px] mt-0.5">
                      {f.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="pt-2">
            <button
              type="submit"
              disabled={uploading}
              className="px-5 py-2.5 bg-[#0A0A0A] text-white text-xs font-semibold rounded-lg hover:bg-black disabled:opacity-50 transition-colors inline-flex items-center"
            >
              {uploading && <RefreshCw className="w-3.5 h-3.5 mr-2 animate-spin" />}
              {uploading ? "Extracting Embeddings & Enrolling..." : "Enroll Identity"}
            </button>
          </div>
        </form>
      </div>

      {/* Enrolled Gallery Grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-[#111827] uppercase tracking-wider">
            Enrolled Gallery Identities ({gallery.length})
          </h2>
          <span className="text-xs text-[#6B7280]">Real-time Prototype Store</span>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-44 bg-[#F9FAFB] border border-[#E5E7EB] rounded animate-pulse" />
            ))}
          </div>
        ) : gallery.length === 0 ? (
          <div className="p-12 border border-[#E5E7EB] rounded text-center bg-[#F9FAFB]">
            <Users className="w-8 h-8 text-[#9CA3AF] mx-auto mb-2" />
            <p className="text-sm font-medium text-[#111827]">Gallery is empty</p>
            <p className="text-xs text-[#6B7280] mt-1">Enroll your first identity above to begin matching.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {gallery.map((item) => (
              <div key={item.identity_id} className="border border-[#E5E7EB] rounded-lg p-4 bg-white flex flex-col justify-between">
                <div>
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="text-xs font-bold text-[#0A0A0A] flex items-center">
                        {item.name}
                        <span className="ml-2 text-[10px] text-[#6B7280] font-mono bg-[#FAFAFA] border border-[#E5E7EB] px-1.5 py-0.2 rounded">
                          ID: {item.identity_id}
                        </span>
                      </div>
                      <div className="text-[11px] text-[#6B7280] mt-0.5 font-mono">
                        {item.num_samples} crops • {item.num_exemplars} exemplars
                      </div>
                    </div>
                    <button
                      onClick={() => setDeleteTarget({ id: item.identity_id, name: item.name })}
                      title="Delete Identity"
                      className="text-[#6B7280] hover:text-[#DC2626] p-1.5 rounded hover:bg-[#F9FAFB] transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Thumbnail Strip */}
                  <div className="flex space-x-1.5 my-3 overflow-x-auto py-1">
                    {item.image_urls && item.image_urls.length > 0 ? (
                      item.image_urls.slice(0, 4).map((url, i) => (
                        <img
                          key={i}
                          src={`${API_BASE}${url}`}
                          alt="crop"
                          className="w-12 h-20 object-cover rounded border border-[#E5E7EB] bg-[#F9FAFB]"
                          onError={(e) => {
                            (e.target as any).style.display = "none";
                          }}
                        />
                      ))
                    ) : (
                      <div className="w-12 h-20 bg-[#F9FAFB] border border-[#E5E7EB] rounded flex items-center justify-center text-[10px] text-[#9CA3AF]">
                        No img
                      </div>
                    )}
                  </div>

                  {/* Adaptive Threshold info */}
                  <div className="space-y-1 text-[11px] border-t border-[#E5E7EB] pt-2 text-[#4B5563]">
                    <div className="flex justify-between">
                      <span>Adaptive Threshold (τᵢ):</span>
                      <span className="font-semibold text-[#111827]">{item.adaptive_tau.toFixed(3)}</span>
                    </div>
                    {item.nearest_lookalike_id && (
                      <div className="flex justify-between items-center text-[#D97706] bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200">
                        <span>Nearest Look-Alike: ID {item.nearest_lookalike_id}</span>
                        <span className="font-medium">{(item.nearest_lookalike_sim * 100).toFixed(1)}% sim</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-[2px] p-4">
          <div className="bg-white rounded-lg border border-[#E5E7EB] shadow-xl max-w-sm w-full p-5 space-y-4 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-full bg-red-50 border border-red-200 flex items-center justify-center flex-shrink-0 text-[#DC2626]">
                <Trash2 className="w-4 h-4" />
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-[#0A0A0A]">Delete Enrolled Identity?</h3>
                <p className="text-xs text-[#6B7280] mt-1.5 leading-relaxed">
                  Are you sure you want to remove <strong className="text-[#0A0A0A] font-semibold">{deleteTarget.name}</strong> (ID: {deleteTarget.id})?
                  The gallery covariance whitening matrix and look-alike thresholds will be automatically re-fitted.
                </p>
              </div>
            </div>

            <div className="flex justify-end space-x-2 pt-3 border-t border-[#E5E7EB]">
              <button
                type="button"
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
                className="px-3 py-1.5 text-xs font-medium text-[#4B5563] hover:text-[#0A0A0A] hover:bg-[#F3F4F6] rounded border border-[#E5E7EB] transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmDelete}
                disabled={deleting}
                className="px-3 py-1.5 text-xs font-medium text-white bg-[#DC2626] hover:bg-[#B91C1C] rounded transition-colors inline-flex items-center space-x-1.5 disabled:opacity-50"
              >
                {deleting && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                <span>{deleting ? "Deleting..." : "Delete Identity"}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

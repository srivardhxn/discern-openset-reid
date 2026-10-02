import React, { useState, useEffect } from "react";
import { Navbar, PageId } from "./components/Navbar";
import { MatchPage } from "./pages/MatchPage";
import { EnrollPage } from "./pages/EnrollPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { LookAlikesPage } from "./pages/LookAlikesPage";
import { DemoScenarioModal } from "./components/DemoScenarioModal";
import { api } from "./api";

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>("match");
  const [enrolledCount, setEnrolledCount] = useState<number>(0);
  const [isDemoOpen, setIsDemoOpen] = useState<boolean>(false);

  useEffect(() => {
    loadEnrolledCount();
  }, []);

  const loadEnrolledCount = async () => {
    try {
      const gallery = await api.getGallery();
      setEnrolledCount(gallery.length);
    } catch {
      // fallback
    }
  };

  return (
    <div className="min-h-screen bg-white text-[#0A0A0A] flex flex-col font-sans antialiased selection:bg-[#0A0A0A] selection:text-white">
      {/* Top Navbar */}
      <Navbar
        activePage={activePage}
        setActivePage={setActivePage}
        enrolledCount={enrolledCount}
        onOpenDemo={() => setIsDemoOpen(true)}
      />

      {/* Main Content Area (Max width 1200px) */}
      <main className="flex-1 px-4 sm:px-6">
        {activePage === "match" && <MatchPage />}
        {activePage === "enroll" && <EnrollPage onGalleryChange={loadEnrolledCount} />}
        {activePage === "evaluation" && <EvaluationPage />}
        {activePage === "lookalikes" && <LookAlikesPage />}
      </main>

      {/* Minimal Footer */}
      <footer className="border-t border-[#E5E7EB] bg-white py-3.5 mt-8 font-sans">
        <div className="max-w-[1200px] mx-auto px-4 sm:px-6 flex flex-col sm:flex-row justify-between items-center text-xs text-[#6B7280] gap-2">
          <div className="flex items-center space-x-2">
            <span className="font-bold text-[#0A0A0A]">Discern</span>
            <span>—</span>
            <span>Open-Set Person Re-Identification Prototype</span>
          </div>
          <div className="flex items-center space-x-3 text-[11px] font-mono">
            <span>FastAPI: :8000</span>
            <span>•</span>
            <span>OSNet x0.5 (0.61M params)</span>
            <span>•</span>
            <span className="text-[#16A34A] font-semibold flex items-center">
              <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A] mr-1" />
              Engine Online
            </span>
          </div>
        </div>
      </footer>

      {/* Global Demo Scenario Modal */}
      <DemoScenarioModal isOpen={isDemoOpen} onClose={() => setIsDemoOpen(false)} />
    </div>
  );
};

export default App;

import React from "react";
import { UserCheck, Users, BarChart3, Eye, Play } from "lucide-react";

export type PageId = "match" | "enroll" | "evaluation" | "lookalikes";

interface NavbarProps {
  activePage: PageId;
  setActivePage: (p: PageId) => void;
  enrolledCount: number;
  onOpenDemo: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activePage,
  setActivePage,
  enrolledCount,
  onOpenDemo,
}) => {
  const navItems: { id: PageId; label: string; icon: React.ReactNode }[] = [
    { id: "match", label: "Match & Verify", icon: <UserCheck className="w-3.5 h-3.5 mr-1.5" /> },
    { id: "enroll", label: "Gallery & Enroll", icon: <Users className="w-3.5 h-3.5 mr-1.5" /> },
    { id: "evaluation", label: "Evaluation & Benchmarks", icon: <BarChart3 className="w-3.5 h-3.5 mr-1.5" /> },
    { id: "lookalikes", label: "Look-Alikes", icon: <Eye className="w-3.5 h-3.5 mr-1.5" /> },
  ];

  return (
    <header className="sticky top-0 z-40 bg-white border-b border-[#E5E7EB] font-sans">
      <div className="max-w-[1200px] mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          {/* Logo & Brand */}
          <div
            className="flex items-center space-x-2.5 cursor-pointer"
            onClick={() => setActivePage("match")}
          >
            <div className="w-7 h-7 rounded-md bg-[#0A0A0A] flex items-center justify-center text-white font-bold text-xs tracking-tight">
              D
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="font-bold text-sm text-[#0A0A0A] tracking-tight">Discern</span>
              <span className="text-[10px] text-[#6B7280] font-mono bg-[#FAFAFA] border border-[#E5E7EB] px-1.5 py-0.2 rounded">
                Open-Set Re-ID
              </span>
            </div>
          </div>

          {/* Navigation Links (4 tabs only) */}
          <nav className="flex space-x-1">
            {navItems.map((item) => {
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActivePage(item.id)}
                  className={`inline-flex items-center px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                    isActive
                      ? "bg-[#FAFAFA] text-[#0A0A0A] border border-[#E5E7EB]"
                      : "text-[#6B7280] hover:text-[#0A0A0A] hover:bg-[#FAFAFA]"
                  }`}
                >
                  {item.icon}
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Right Header Actions */}
          <div className="flex items-center space-x-2.5 text-xs">
            {/* Gallery Status Badge */}
            <div className="hidden sm:flex items-center px-2.5 py-1 rounded-md bg-[#FAFAFA] border border-[#E5E7EB] text-[#0A0A0A] font-mono text-[11px]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A] mr-1.5" />
              <span>{enrolledCount} Enrolled</span>
            </div>

            {/* Run Demo Scenario Button */}
            <button
              onClick={onOpenDemo}
              className="inline-flex items-center px-3 py-1.5 bg-[#0A0A0A] text-white text-xs font-semibold rounded-md hover:bg-black transition-colors"
            >
              <Play className="w-3 h-3 mr-1.5 fill-current" />
              Demo
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

"use client";

import GlobalSidebar, { type GlobalTab } from "./GlobalSidebar";

interface AppShellProps {
  activeTab: GlobalTab;
  onTabChange: (tab: GlobalTab) => void;
  children: React.ReactNode;
}

export default function AppShell({ activeTab, onTabChange, children }: AppShellProps) {
  return (
    <div className="flex h-full w-full">
      <GlobalSidebar activeTab={activeTab} onTabChange={onTabChange} />
      {/* pb-14 leaves a 56px gutter at the bottom of the scrollable area
          so the (fixed-mounted) status footer never covers the last line. */}
      <div className="flex-1 overflow-y-auto pb-14">{children}</div>
    </div>
  );
}

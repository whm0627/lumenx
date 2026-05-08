"use client";

import GlobalSidebar, { type GlobalTab } from "./GlobalSidebar";

interface AppShellProps {
  activeTab: GlobalTab;
  onTabChange: (tab: GlobalTab) => void;
  children: React.ReactNode;
}

// pb-14 on the shell itself leaves a 56px gutter at the bottom of both
// columns so the full-width fixed status footer never covers the sidebar's
// v0.1.0 row or the main content's last line.
export default function AppShell({ activeTab, onTabChange, children }: AppShellProps) {
  return (
    <div className="flex h-full w-full pb-14">
      <GlobalSidebar activeTab={activeTab} onTabChange={onTabChange} />
      <div className="flex-1 overflow-y-auto">{children}</div>
    </div>
  );
}

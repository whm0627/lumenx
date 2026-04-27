import "./globals.css";
import EnvConfigChecker from "@/components/EnvConfigChecker";
import GlobalStatusFooter from "@/components/layout/GlobalStatusFooter";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <title>LumenX Studio</title>
        <meta name="description" content="AI-Native Motion Comic Creation Platform" />
      </head>
      <body className="font-sans bg-background text-foreground antialiased">
        <EnvConfigChecker />
        {children}
        {/* Mounted at root so navigation between AppShell and full-screen
            views (project / series detail) doesn't unmount it. Uses fixed
            positioning so it sits at the bottom regardless of inner flex
            layouts. */}
        <GlobalStatusFooter />
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Language Evolution Simulator",
  description: "Create a civilization and the language its people speak.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="top-nav">
          <Link href="/" className="brand">
            Language Evolution
          </Link>
          <div className="nav-links">
            <Link href="/">New language</Link>
            <Link href="/languages">My languages</Link>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}

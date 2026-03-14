import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "COVID-19 vs Pneumonia Classifier",
  description: "Hybrid CNN-Transformer model for detecting COVID-19 and Pneumonia from X-ray images",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
            <h1 className="text-2xl font-bold text-indigo-900">
              COVID-19 vs Pneumonia Classifier
            </h1>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
          {children}
        </main>
        <footer className="bg-white border-t mt-auto">
          <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
            <p className="text-center text-gray-500 text-sm">
              Powered by Hybrid CNN-Transformer Model
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}

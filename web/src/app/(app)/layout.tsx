import { Footer } from "@/components/general/Footer";
import { Navbar } from "@/components/general/Navbar";


export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
            <Navbar/>
            {children}
            <Footer/>
      </body>
    </html>
  );
}

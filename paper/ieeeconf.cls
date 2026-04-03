\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ieeeconf}[2026/04/01 Minimal local IEEE conference wrapper]

\DeclareOption{conference}{}
\DeclareOption{letterpaper}{}
\DeclareOption{10pt}{}
\DeclareOption*{}
\ProcessOptions\relax
\LoadClass[10pt,twocolumn]{article}

\RequirePackage[letterpaper,margin=0.72in]{geometry}
\setlength{\columnsep}{0.24in}
\setlength{\parindent}{1em}

\newcommand{\IEEEoverridecommandlockouts}{}
\newcommand{\overrideIEEEmargins}{}

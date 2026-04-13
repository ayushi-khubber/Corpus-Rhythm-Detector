# Corpus Rhythm Detector

An NLP-powered system that analyzes how word frequencies evolve across novels over time, transforming literary texts into interactive time-series visualizations.

## Overview

The **Corpus Rhythm Detector** treats novels as linguistic time series, dividing them into sequential segments and tracking word frequency patterns to extract meaningful insights. Instead of analyzing text as a single block, this system reveals how themes—represented by specific words—rise, fall, and evolve from a book's beginning to its end.

**Supported novels include:**
- *Moby Dick*
- *War and Peace*
- *Crime and Punishment*
- *The Great Gatsby*
- *...and any text file you upload!*

## Research Foundation

This project adapts the concept of **lexical frequency series** introduced by Professors Adam Pawłowski and Tomasz Walkowiak at the University of Wrocław. Their work demonstrated how tracking word frequencies across large text datasets reveals meaningful linguistic patterns—an idea here applied to literary analysis.

## Features

### Backend (Python)
- **Text Cleaning & Tokenization** – Regex-based word extraction
- **Intelligent Segmentation** – Divides novels into ~80 equal parts
- **Normalized Frequency Calculation** – `(word count / total words) × 10,000`
- **Pattern Detection** – Peak identification, regression trends, statistical correlations
- **JSON Export** – Structured data ready for visualization

### Frontend (HTML/CSS/JS + Chart.js)
- **Interactive Line Graphs** – Plot word frequencies across narrative progression
- **Multi-word Comparison** – Compare multiple words side-by-side
- **Segment Explorer** – Click any point on the graph to view the original text excerpt
- **Custom Upload** – Upload `analysis.json` or raw `.txt` files for real-time processing
- **Auto-generated Insights** – Detected peaks, trends, and word correlations

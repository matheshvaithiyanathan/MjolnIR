# MjölnIR

**MjÖlnIR** is a user-friendly tool for visualizing and analyzing two-dimensional datasets without the need to write repetitive Python or MATLAB code. It’s designed to simplify your workflow by letting you inspect, process, and export results quickly while keeping your work organized as one project. The goal is to reduce complexity so your data isn’t just seen, but clearly understood. Why this name? It is just unique and ends it ends IR :) Many other names I considered were already in use. 
To download the program, as .exe file, visit [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15874435.svg)](https://doi.org/10.5281/zenodo.15874435) [![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/CPM2vY8x) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/matheshvaithiyanathan)

---

## 📌 Scope

Originally built for 2D infrared (2DIR) spectroscopy, MjÖlnir is versatile enough to handle a wide range of other 2D datasets. Whether you're working with excitation–emission spectrum, 2D electronic spectra, or any other dataset arranged in an X–Y–Z format, this tool adapts effortlessly to your needs. A picture of the window is given below 

![Description of the image](image.png)

---

## ⚙️ Analytical Features

Using Laplacian, it can identify peaks in noisy or saturated datasets. Sensitivity can be adjusted, giving you precise control over the detection process.

For deeper analysis, the tool makes it easy to extract and examine cross-sections along both the x and y axes. You can fit these slices interactively with just mouse draging and overlay multiple cross-sections for direct, side-by-side comparison.

Background artifacts are handled automatically through spline-based baseline correction. To make sure the correction meets your standards, a visual preview is provided.

When it’s time to present your results, MjÖlnIR offers one-click export options in vector formats like SVG, PNG or the data itself. You can customize axis labels, color scales, and gridlines to match your expectation, ensuring your figures are ready for publication with less formatting work.

---

## 📥 Data Import Format

To keep the analysis consistent and avoid ambiguity, MjÖlnIR works with 2D spectral data saved in a standardized matrix format. In this setup, the first row should contain your probe (X-axis) values, while the first column contains the pump (Y-axis) values. The remaining cells hold the Z-data aligned to these dimensions and with no headers.

For reliable interpretation across different systems and regional settings, files must use commas `,` as delimiters and periods `.` as decimal points. For example, a valid data point would look like `1950.5`. The file can be a .csv or .txt file. 

This clear and simple format ensures that your data loads correctly every time, with no guesswork about axis assignments or number formatting. 


---
## ✏️ Author's Note

This python project was developed duing my PhD work in AG Horch, FU Berlin, funded by the DFG UniSysCat grant (EXC 2008 – 390540038). It began as a custom Matplotlib-based interactive plotting tool and was later transformed into a PyQt-based application with the assistance of AI tools like Gemini 1.5 Flash and DeepSeek-R1. It was a personal application to simplify my own data 2DIR analysis, and now it is available for other users.

For citation purposes, please use the following DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15874435.svg)](https://doi.org/10.5281/zenodo.15874435)

---

## 📬 Contact

I have included the functions that were useful for me with 2D-IR data. If you have any questions, suggestions, or improvements for MjÖlnir, feel free to reach out at www.linkedin.com/in/matheshvaithiyanathan


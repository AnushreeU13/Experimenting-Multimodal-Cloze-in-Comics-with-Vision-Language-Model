# Multimodal Cloze in Comics: Teaching AI to "Read Between the Panels"

> *"Comics are a unique narrative medium that convey stories through the interaction of sequential images and text... unlike purely textual narratives, much of the storytelling in comics occurs implicitly."*

This project explores **Narrative Closure** in Artificial Intelligence: the ability to infer missing actions, dialogue, or events that happen in the "gutter"—the space between comic panels. Just as humans mentally bridge the gap between two static images to imagine the motion connecting them, we test whether modern Multimodal Large Language Models (MLLMs) can look at a sequence of 5 comic panels and **predict exactly what happens next**.

## 📄 The Challenge
Based on the foundational work *The Amazing Mysteries of the Gutter* (**Iyyer et al., 2017**), we frame this as a **Multimodal Cloze Task**.
Given context panels $P_1...P_5$ (images + dialogue), the model must:
1.  **Predict** the narrative content (dialogue, action, setting) of the hidden target panel $P_6$.
2.  **Describe** the visual scene of $P_6$.
3.  **Generate** the missing panel image using Stable Diffusion.

## 🏗️ Architecture & Approach
We moved beyond early LSTM+CNN approaches to leverage state-of-the-art Large Vision-Language Models (VLMs).

### 1. The Model: LLaVA-OneVision
We selected **LLaVA-OneVision-Qwen2-7B** as our backbone.
*   **Why?** Most VLMs (like LLaVA-1.5) have a small context window (~4k tokens). A sequence of 5 images + dialogue requires ~4,000 tokens alone.
*   **Solution**: LLaVA-OneVision's **32k context window** allowed us to feed the entire 5-panel visual narrative without truncation.
*   **Training**: Fine-tuned using **QLoRA** (4-bit quantization) on NVIDIA H200 GPUs.

### 2. "Silver" Ground Truth with Gemini
The original COMICS dataset (1930s-50s) had noisy OCR and no visual descriptions. We created a high-quality "Silver Standard" dataset using **Google Gemini 2.5 Flash**:
*   **Paradigm A (Prediction)**: We showed Gemini panels 1-5 and asked it to *predict* panel 6 (aligns with inference task).
*   **Paradigm B (Description)**: We showed Gemini all 6 panels and asked it to *describe* panel 6 (visual ground truth).
*   **Result**: We generated ~300k high-quality training examples.

### 3. Visualizing the Imagination
Finally, we fed the model's textual predictions into **Stable Diffusion v1.5** to generate the actual image of the next panel, effectively "drawing" the predicted future.

## 📊 Key Results
Our fine-tuned models demonstrated a significant leap in understanding narrative continuity:
*   **+111% Improvement** in ROUGE-2 (phrase matching) over zero-shot baselines.
*   **Character Consistency**: The model successfully tracked characters (e.g., "Wing", "Barbara") and their motivations across panels.
*   **Narrative Flow**: Unlike zero-shot models that hallucinated generic scenes, our model referenced specific plot points from earlier panels (e.g., "the hardest fight lies ahead").

## 📂 Repository Structure
*   `01_data_pipeline.ipynb`: Data ingestion and comic-level splitting (Train/Val/Test).
*   `02_data_cleaning...`: Scripts for Zero-Shot evaluation on clusters.
*   `03-06_*.ipynb`: Evaluation notebooks comparing Zero-Shot vs. Fine-Tuned models on Prediction and Description tasks.
*   `07-08_generate...`: Pipeline to generating "Silver" labels using Vertex AI (Gemini).
*   `09_generate_comic_panel.py`: Stable Diffusion pipeline for generating the final images.

## 🛠️ Credits
**PanelPioneers Team**:
*   **Anushree Udhayakumar**: LLaVA experimentation, Stable Diffusion pipeline, Report Lead.
*   **Harshvardhan Sekar**: GCS Data orchestration, Gemini Silver labeling pipeline, Fine-tuning orchestration.
*   **Prisha Singhania**: Evaluation of OpenFlamingo/QWEN architectures.
*   **Sonia Navale**: Literature survey and model parameter selection.

*Dataset provided by [The Amazing Mysteries of the Gutter](https://github.com/miyyer/comics) (Iyyer et al., CVPR 2017).*

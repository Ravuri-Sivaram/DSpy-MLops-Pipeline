# 🚀 DSPy-MLOps: Automated Programmatic Extraction Pipeline

### **Project Description**
This project is an automated, **domain-agnostic MLOps pipeline** designed to extract structured insights from complex technical documents. While the underlying infrastructure is built to handle any specialized data domain, it is currently demonstrated through the extraction of **Blockchain Consensus Layer Security** vulnerabilities. By leveraging the **DSPy framework** and **Gemini 1.5 Flash**, it replaces fragile manual prompt engineering with a programmatic optimization loop and an automated CI/CD performance gatekeeper.

---

### **🏗️ System Architecture**
The pipeline follows a **Continuous Evaluation (CE)** workflow, offloading heavy compute to the cloud to maintain a lightweight local footprint:

1.  **Data Ingestion:** Technical abstracts are stored in a structured `dataset.json` (22 diverse samples).
2.  **Programmatic Signature:** Defines the extraction logic using DSPy Signatures (`text_input -> structured_data_output`).
3.  **Optimization Loop:** Utilizes `BootstrapFewShot` to automatically select and "bootstrap" the most effective few-shot demonstrations for the LLM.
4.  **Performance Gatekeeper:** A CI/CD gate implemented via **GitHub Actions** that executes the optimized model against a validation benchmark on every push.
5.  **Artifact Deployment:** If the score exceeds the **75% threshold**, the optimized model weights (`compiled_extractor.json`) are saved and uploaded as a production-ready artifact.

---

### **🛠️ Tech Stack**
* **Framework:** DSPy (Declarative Self-improving Language Programs)
* **LLM Engine:** Google Gemini 1.5 Flash (via LiteLLM)
* **Automation/CI/CD:** GitHub Actions
* **Data Management:** JSON-based "Golden Dataset"
* **Environment:** WSL2 (Windows Subsystem for Linux), Python 3.12

---

### **💡 Why This Project?**
Traditional LLM extraction relies on "vibes-based" manual prompting, which is fragile, unversioned, and hard to scale. This project demonstrates a **production-first approach** to AI:
* **Reliability:** We move from "guessing" if a prompt works to "proving" its effectiveness via quantitative benchmarks.
* **Cost Efficiency:** Using DSPy allows for "Pro-tier" extraction accuracy while utilizing faster, more cost-effective "Flash-tier" models.
* **Automation:** The system "self-heals"—whenever the dataset is updated, the pipeline automatically re-optimizes the prompt to handle the new data variations.

---

### **🎯 Demonstrating Core Skills**
* **MLOps Architecture:** Implementing automated quality gates and performance thresholds in a CI/CD environment.
* **Prompt Programming:** Leveraging data-driven optimization over manual trial-and-error prompting.
* **Software Engineering:** Secure API management using GitHub Secrets and modular, reusable Python logic.
* **Specialized Domain Application:** Demonstrated ability to apply AI frameworks to analyze complex technical domains (e.g., Blockchain MEV, Eclipse Attacks, and Liveness threats).

---

### **🚀 How to Run**

#### **1. Clone the Repository**
```bash
git clone [https://github.com/Ravuri-Sivaram/DSpy-MLops-Pipeline.git](https://github.com/Ravuri-Sivaram/DSpy-MLops-Pipeline.git)
cd DSpy-MLops-Pipeline


### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Set Up API Key**
```bash
export GEMINI_API_KEY="your_google_ai_studio_key"
```

### **4. Execute Optimization**
```bash
python optimizer.py
```

---

## 🔮 Future Development (Version 2)

- **Data Scale-Up**  
  Expanding the "Golden Dataset" to 100+ samples for improved edge-case coverage.

- **Bayesian Optimization**  
  Transitioning to the MIPROv2 optimizer for joint instruction and example optimization.

- **Containerization & Serving**  
  Packaging the extractor into a Docker container and exposing a FastAPI endpoint for microservice deployment.

- **Semantic Metrics**  
  Implementing BERTScore or Cosine Similarity for more nuanced output validation.
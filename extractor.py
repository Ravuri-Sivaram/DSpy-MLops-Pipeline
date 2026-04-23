import dspy
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Sanity check to ensure the .env file is actually being read
if not api_key:
    print("WARNING: API Key not found. Check that your file is named exactly '.env' and has no txt extension.")

# 2. Configure the Language Model via API
lm = dspy.LM('gemini/gemini-2.5-flash', api_key=api_key)
dspy.configure(lm=lm)

# 3. Define the Programmatic Signature
class BlockchainVulnerabilityExtractor(dspy.Signature):
    """Extracts security vulnerabilities, consensus mechanisms, and countermeasures from academic abstracts."""
    
    paper_text = dspy.InputField(desc="The abstract or methodology section of the research paper.")
    
    consensus_mechanism = dspy.OutputField(desc="The specific consensus protocol discussed (e.g., PoW, PoS, PBFT).")
    vulnerability_category = dspy.OutputField(desc="The type of attack or vulnerability (e.g., MEV, Sybil, Eclipse).")
    proposed_countermeasure = dspy.OutputField(desc="A brief 1-sentence summary of the proposed defense.")

# 4. Initialize the DSPy Module
extractor = dspy.Predict(BlockchainVulnerabilityExtractor)
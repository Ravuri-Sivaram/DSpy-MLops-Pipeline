import dspy
import json
import sys  # Added for the exit logic
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate # Added for final validation
from extractor import BlockchainVulnerabilityExtractor, lm

# 1. Connect DSPy to the model already configured in extractor.py
dspy.configure(lm=lm)

# 2. Define the Metric Function
def vulnerability_metric(example, pred, trace=None):
    """Evaluates how perfectly the LLM extracted the data."""
    score = 0.0
    # Use .get() to prevent crashes if the model misses a field
    pred_consensus = getattr(pred, 'consensus_mechanism', "").lower()
    pred_vulnerability = getattr(pred, 'vulnerability_category', "").lower()
    
    if example.consensus_mechanism.lower() in pred_consensus:
        score += 0.5
    if example.vulnerability_category.lower() in pred_vulnerability:
        score += 0.5
    return score

# 3. Load the Golden Dataset
def load_data():
    with open("data/dataset.json", "r") as f:
        raw_data = json.load(f)
    
    dataset = []
    for item in raw_data:
        example = dspy.Example(
            paper_text=item["paper_text"],
            consensus_mechanism=item["consensus_mechanism"],
            vulnerability_category=item["vulnerability_category"],
            proposed_countermeasure=item["proposed_countermeasure"]
        ).with_inputs("paper_text")
        dataset.append(example)
    return dataset

trainset = load_data()

# 4. Initialize Baseline Module
baseline_extractor = dspy.Predict(BlockchainVulnerabilityExtractor)

# 5. Configure Optimizer
optimizer = BootstrapFewShot(
    metric=vulnerability_metric,
    max_bootstrapped_demos=2, 
    max_labeled_demos=2
)

print("Starting DSPy Compilation Phase...")

# 6. Compile the Program
compiled_extractor = optimizer.compile(
    student=baseline_extractor, 
    trainset=trainset
)

# --- GATEKEEPER LOGIC ---

print("\nEvaluating Optimized Model against Threshold...")

# 7. Final evaluation
# In a real scenario, you'd use a 'testset' separate from 'trainset'
evaluator = Evaluate(devset=trainset, num_threads=1, display_progress=True)
final_score = evaluator(compiled_extractor, metric=vulnerability_metric)

THRESHOLD = 0.75 # Requirement: At least 75% accuracy to pass to production
print(f"Final Score: {final_score} | Threshold: {THRESHOLD}")

# 8. Rejection Logic
if final_score < THRESHOLD:
    print("❌ REJECTED: Model performance below threshold. Not saving artifact.")
    sys.exit(1) # Kills the GitHub Action and prevents the upload step
else:
    print("✅ PASSED: Model quality standards met.")
    compiled_extractor.save("compiled_extractor.json")
    print("Saved to compiled_extractor.json")
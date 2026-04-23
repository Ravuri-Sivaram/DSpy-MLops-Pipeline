import dspy
import json
from dspy.teleprompt import BootstrapFewShot
from extractor import BlockchainVulnerabilityExtractor, lm

# 1. Connect DSPy to the model already configured in extractor.py
dspy.configure(lm=lm)

# 2. Define the Metric Function
def vulnerability_metric(example, pred, trace=None):
    """Evaluates how perfectly the LLM extracted the data."""
    score = 0.0
    if example.consensus_mechanism.lower() in pred.consensus_mechanism.lower():
        score += 0.5
    if example.vulnerability_category.lower() in pred.vulnerability_category.lower():
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

print("Starting DSPy Compilation Phase... (This may take a minute)")

# 6. Compile the Program
compiled_extractor = optimizer.compile(
    student=baseline_extractor, 
    trainset=trainset
)

print("\nCompilation Complete! Saving optimal weights...")

# 7. Serialize the Model Artifact
compiled_extractor.save("compiled_extractor.json")
print("Saved to compiled_extractor.json")
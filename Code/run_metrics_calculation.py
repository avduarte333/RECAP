

from extraction_utils import MetricsCalculationTask

# Example usage: Calculate metrics from an extraction JSON file
task = MetricsCalculationTask(
    extraction_json_path="./Results/The_Count_of_Monte_Cristo_-_Alexandre_Dumas/Extractions/The_Count_of_Monte_Cristo_-_Alexandre_Dumas_extraction_deepseek-chat_feedback_gpt-4.1-2025-04-14.json",
    min_tokens=40,                    # Minimum tokens for contiguous spans
    max_mismatch_tokens=5            # Maximum mismatch tokens for span merging
)

# Execute the metrics calculation task
task.run()

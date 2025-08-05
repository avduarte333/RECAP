from extraction_utils import BookExtractionTask

task = BookExtractionTask(
    json_file_path="Adventures_of_Huckleberry_Finn_-_Mark_Twain/Adventures_of_Huckleberry_Finn_-_Mark_Twain_summary_gemini-2.5-pro-exp-03-25.json",
    model_name="gpt-4.1",                    # Target model for extractions
    evaluation_model_name="gemini-2.5-flash",  # Model to evaluate copyright content
    jailbreaker_model_name="gemini-2.5-flash", # Model for jailbreak prompt generation
    feedback_model_name="gpt-4.1",          # Model for feedback loops
    results_base_folder="./Results"                     # Base folder to save results
)

task.run()

# output_handler.py
def save_summary(summary_data: dict, output_path: str) -> None:
    print(f"Saving summary to {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Document Summary\n\n")
            f.write("## Detailed Summary\n\n")
            f.write(f"{summary_data['detailed_summary']}\n\n")
            f.write("## Key Points\n\n")
            f.write(f"{summary_data['key_points']}\n")
        print("Summary saved successfully")
    except Exception as e:
        print(f"Error saving summary to {output_path}: {e}")
        raise

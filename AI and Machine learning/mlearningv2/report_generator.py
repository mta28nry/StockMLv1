# mlearning/report_generator.py

"""
Module: report_generator.py
Purpose:
- Summarize combo performance from DB
- Group by target
- Print to console + save to CSV
"""

import os
import sqlite3

import pandas as pd

from config import DB_PATH, OUTPUT_DIR, create_logger

report_generator = create_logger("report_generator", log_to_file=True)





def generate_report(top_n=3, output_csv=True):
    report_generator.info("📊 Generating training report from meta DB...")
    conn = sqlite3.connect(DB_PATH)

    try:
        df = pd.read_sql("SELECT * FROM meta ORDER BY date_trained DESC", conn)
        if df.empty:
            report_generator.error("⚠️ No data found in meta table.")
            return

        summary_rows = []
        for target in df['target'].unique():
            df_t = df[df['target'] == target]
            top_df = df_t.sort_values(["accuracy", "shap_quality"], ascending=False).head(top_n)
            print(f"\n🔹 Top {top_n} strategies for {target}:\n")
            print(top_df[["combo", "accuracy", "shap_quality"]].to_string(index=False))

            for _, row in top_df.iterrows():
                summary_rows.append({
                    "target": target,
                    "combo": row.combo,
                    "accuracy": row.accuracy,
                    "shap_quality": row.shap_quality,
                    "kept": row.kept,
                    "timestamp": row.timestamp
                })

        if output_csv:
            summary_df = pd.DataFrame(summary_rows)
            out_path = os.path.join(OUTPUT_DIR, "report_summary.csv")
            summary_df.to_csv(out_path, index=False)
            report_generator.info(f"✅ Report saved to: {out_path}")

    except Exception as e:
        report_generator.error(f"❌ Report generation failed: {e}")
    finally:
        conn.close()



def generate_run_readme(summary_dict):
    from config import OUTPUT_DIR
    import os

    if not summary_dict:
        report_generator.error("❌ generate_run_readme() received None. Skipping README generation.")
        return

    try:
        readme_lines = [
            "# 🧠 Run Summary",
            "",
            f"**Run ID:** {summary_dict.get('run_id', 'N/A')}",
            f"**Date:** {summary_dict.get('date', 'N/A')}",
            f"**Dataset rows:** {summary_dict.get('rows', 'N/A')}",
            f"**Best Accuracy:** {summary_dict.get('best_accuracy', 'N/A')}",
        ]
        readme_path = os.path.join(OUTPUT_DIR, "RUN_SUMMARY.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("\n".join(readme_lines))
        report_generator.info(f"✅ Generated summary at {readme_path}")
    except Exception as e:
        report_generator.error(f"❌ Failed to write README: {e}")



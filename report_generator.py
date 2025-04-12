# mlearning/report_generator.py

"""
Module: report_generator.py
Purpose:
- Summarize combo performance from DB
- Group by target
- Print to console + save to CSV
"""

import sqlite3
import pandas as pd
import os
import logging
from config import DB_PATH, OUTPUT_DIR
import datetime
def generate_report(top_n=3, output_csv=True):
    logging.info("📊 Generating training report from meta DB...")
    conn = sqlite3.connect(DB_PATH)

    try:
        df = pd.read_sql("SELECT * FROM meta ORDER BY timestamp DESC", conn)
        if df.empty:
            logging.warning("⚠️ No data found in meta table.")
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
            logging.info(f"✅ Report saved to: {out_path}")

    except Exception as e:
        logging.error(f"❌ Report generation failed: {e}")
    finally:
        conn.close()



def generate_run_readme(meta_summary_df):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    readme_path = os.path.join(OUTPUT_DIR, "README.md")

    lines = [
        f"# 📘 ML Training Summary - {timestamp}\n",
        f"**Dataset rows:** {meta_summary_df.get('rows', 'N/A')}",
        f"**Generated Models:** {meta_summary_df.get('models', 'N/A')}",
        f"**Combos Trained:** {meta_summary_df.get('combos', 'N/A')}",
        f"**Feedback Scores Logged:** {meta_summary_df.get('feedback', 'N/A')}",
        f"**Top Accuracy:** {meta_summary_df.get('top_accuracy', 'N/A'):.4f}",
        f"**Top SHAP Score:** {meta_summary_df.get('top_shap', 'N/A'):.4f}",
        "\n## 🔝 Top Combos:\n"
    ]

    if 'top_df' in meta_summary_df:
        top_df = meta_summary_df["top_df"]
        lines += [top_df.to_markdown(index=False)]

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logging.info(f"📘 README generated: {readme_path}")
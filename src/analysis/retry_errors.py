"""Retry failed ECBSS pairs with reduced concurrency and backoff."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from cbias.scoring.openrouter_client import OpenRouterClient
from analysis.data_loader import BIAS_FAMILY_DESCRIPTIONS
from analysis.ecbss_scorer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, DIRECT_OUTPUT, DIRECT_RAW_OUTPUT

RAW = ROOT / "src/review_stages/analysis_outputs/ecbss_direct_raw.jsonl"

def _retry_single(client, emotion, family_key, family_description, retries=5):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        emotion=emotion,
        family_name=family_key.replace("_", " ").title(),
        family_description=family_description[:600],
    )
    for attempt in range(retries):
        try:
            result = client.chat_json(SYSTEM_PROMPT, user_prompt, temperature=0.0)
            assert "ecbss" in result
            return {
                "emotion": emotion, "family_key": family_key,
                "ecbss": int(np.clip(result["ecbss"], -1000, 1000)),
                "direction": result.get("direction", "unknown"),
                "confidence": result.get("confidence", 50),
                "rationale": result.get("rationale", ""),
                "status": "ok",
            }
        except Exception as e:
            wait = 4 * (2 ** attempt)
            print(f"    retry {attempt+1}/{retries} for {emotion}×{family_key[:30]}: {e!s:.60s} — wait {wait}s")
            time.sleep(wait)
    return {"emotion": emotion, "family_key": family_key, "ecbss": 0,
            "direction": "neutral", "confidence": 0, "rationale": "retry failed", "status": "error"}


def run_retry(max_workers: int = 8):
    lines = RAW.read_text().splitlines()
    records = [json.loads(l) for l in lines]
    failed = [r for r in records if r["status"] == "error"]
    ok     = [r for r in records if r["status"] == "ok"]
    print(f"Retrying {len(failed)} failed pairs with {max_workers} workers …")

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model   = os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
    client  = OpenRouterClient(api_key=api_key, model=model, timeout_seconds=120)

    retried = []
    total = len(failed)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_retry_single, client, r["emotion"], r["family_key"],
                        BIAS_FAMILY_DESCRIPTIONS.get(r["family_key"], r["family_key"])): r
            for r in failed
        }
        done = 0
        for future in as_completed(futures):
            result = future.result()
            retried.append(result)
            done += 1
            if done % 50 == 0:
                still_err = sum(1 for x in retried if x["status"] == "error")
                print(f"  {done}/{total} | errors so far: {still_err}")

    still_errors = sum(1 for r in retried if r["status"] == "error")
    print(f"Retry complete: {total - still_errors}/{total} recovered, {still_errors} still failed")

    # Merge and save raw JSONL
    all_records = ok + retried
    with open(RAW, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # Re-pivot to matrix CSV
    raw_df = pd.DataFrame(all_records)
    pivot = raw_df.pivot_table(index="emotion", columns="family_key", values="ecbss", aggfunc="first")
    pivot.columns.name = None
    pivot.index.name = "emotion"
    # Fill remaining errors with 0
    pivot = pivot.fillna(0)
    pivot.to_csv(DIRECT_OUTPUT)
    print(f"Saved updated matrix: {pivot.shape}")
    return pivot

if __name__ == "__main__":
    run_retry(max_workers=8)

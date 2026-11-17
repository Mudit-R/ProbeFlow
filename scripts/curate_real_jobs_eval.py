"""Record author judgments for the specific 2026-11-17 job snapshot.

Indices were manually reviewed against titles and locations. Each positive is
one reasonable match, not an exhaustive list or eligibility determination.
Review the generated IDs and snapshot hash before using with any other data.
"""
import hashlib
import json
from pathlib import Path

SNAPSHOT = Path("artifacts/real_jobs_snapshot.json")
OUTPUT = Path("fixtures/real_job_queries.json")
CASES = [
    ("software engineering intern New York", [88, 228]),
    ("software engineer intern London", [221, 72]),
    ("internship software engineer Denver", [143, 147]),
    ("software engineer internship Palo Alto", [149, 89]),
    ("software engineering internship Singapore", [236, 371]),
    ("software engineering internship Vietnam", [343, 336]),
    ("infrastructure engineering internship New York", [69, 46]),
    ("production infrastructure software internship", [28, 46]),
    ("query engines software internship", [188, 54]),
    ("software engineer new graduate London", [8, 186]),
    ("new graduate production infrastructure engineer", [11, 169]),
    ("new graduate software engineer", [219, 113]),
    ("forward deployed software engineer internship", [209, 20]),
    ("forward deployed software internship Chicago", [129, 240]),
    ("forward deployed software internship commercial", [132, 24]),
    ("forward deployed software internship government", [74, 99]),
    ("software engineer internship defense", [58, 246]),
    ("product design internship", [167, 214]),
    ("product designer internship New York", [214, 47]),
    ("security or privacy engineering internship or new grad", [17, 208]),
    ("software engineer intern Asia", [250, 371]),
    ("software engineer internship United States", [342, 309]),
    ("site reliability or infrastructure intern Singapore", [301, 371]),
    ("software quality or IT support intern", [364, 303]),
    ("data analyst internship Singapore", [368, 329]),
]


def main():
    content = SNAPSHOT.read_bytes()
    data = json.loads(content)
    if len(data["jobs"]) != 375 or data["collected_at_utc"] != "2026-11-17T14:36:15+00:00":
        raise SystemExit("This curation is only for the original manually reviewed snapshot")
    cases = []
    for number, (query, indexes) in enumerate(CASES, 1):
        jobs = [data["jobs"][i] for i in indexes]
        cases.append({"id": f"real-{number:02}", "query": query,
            "relevance": {item["id"]: 1 for item in jobs},
            "judged_jobs": [{"id": item["id"], "title": item["title"],
                              "company": item["company"], "location": item["location"]} for item in jobs]})
    labels = {"snapshot_sha256": hashlib.sha256(content).hexdigest(),
              "method": "25 author-written queries with two author-selected possible positive jobs per query, based on title and location. Positive-only, incomplete judgments. No user authorization or eligibility judgment.",
              "cases": cases}
    OUTPUT.write_text(json.dumps(labels, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Wrote {len(cases)} queries and {sum(len(c['relevance']) for c in cases)} judgments to {OUTPUT}")


if __name__ == "__main__":
    main()


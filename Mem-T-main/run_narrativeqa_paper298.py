"""Run the 298-QA NarrativeQA subset while building memory once per document."""

from collections import OrderedDict

import run_benchmark_local as benchmark


_load_narrativeqa_rows = benchmark.load_narrativeqa_custom


def load_grouped_narrativeqa(*args, **kwargs):
    dataset_path = args[0] if args else kwargs.get("filepath", "")
    original_load_dataset = benchmark.load_dataset

    def load_test_only(path, *load_args, **load_kwargs):
        if path == dataset_path:
            raise ValueError("Use the local test parquet files directly")
        dataset = original_load_dataset(path, *load_args, **load_kwargs)
        if path == "parquet" and "test" in dataset:
            dataset["test"] = dataset["test"].select(range(298))
        return dataset

    benchmark.load_dataset = load_test_only
    try:
        rows = _load_narrativeqa_rows(*args, **kwargs)[:298]
    finally:
        benchmark.load_dataset = original_load_dataset
    documents = OrderedDict()

    for row in rows:
        qa = row["qa"][0]
        document_id = qa["sample_id"].rsplit("_", 1)[0]
        qa["sample_id"] = document_id

        if document_id not in documents:
            for session in row["conversation"]:
                session["metadata"]["sample_id"] = document_id
            documents[document_id] = {
                "qa": [],
                "conversation": row["conversation"],
            }
        documents[document_id]["qa"].append(qa)

    grouped = list(documents.values())
    print(
        f"narrativeqa_subset documents={len(grouped)} "
        f"qa={sum(len(sample['qa']) for sample in grouped)}",
        flush=True,
    )
    return grouped


benchmark.load_narrativeqa_custom = load_grouped_narrativeqa


if __name__ == "__main__":
    benchmark.main()

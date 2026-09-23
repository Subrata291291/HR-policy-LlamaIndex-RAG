from app.vectorstore.pinecone_store import (
    create_pinecone_client,
    get_index,
)

from app.retrieval.retriever import retrieve


TEST_QUERIES = [
    # Supported
    {
        "query": "What is the annual leave entitlement?",
        "expected_source": "Leave_Policy.pdf",
        "supported": True,
    },
    {
        "query": "How many sick leave days are provided?",
        "expected_source": "Leave_Policy.pdf",
        "supported": True,
    },
    {
        "query": "How many days can employees work from home?",
        "expected_source": "Work_From_Home_Policy.pdf",
        "supported": True,
    },
    {
        "query": "What is the internet allowance?",
        "expected_source": "Employee_Benefits.pdf",
        "supported": True,
    },
    {
        "query": "What are the standard working hours?",
        "expected_source": "Attendance_Policy.pdf",
        "supported": True,
    },
    {
        "query": "How much can employees claim for meals during travel?",
        "expected_source": "Travel_Policy.pdf",
        "supported": True,
    },
    {
        "query": "What types of leave are available?",
        "expected_source": "Leave_Policy.pdf",
        "supported": True,
    },
    {
        "query": "How much internet expense is covered?",
        "expected_source": "Employee_Benefits.pdf",
        "supported": True,
    },

    # Unsupported
    {
        "query": "Does the company provide maternity leave?",
        "expected_source": "Leave_Policy.pdf",
        "supported": False,
    },
    {
        "query": "Does the company provide paternity leave?",
        "expected_source": "Leave_Policy.pdf",
        "supported": False,
    },
    {
        "query": "Does the company provide employee bonuses?",
        "expected_source": None,
        "supported": False,
    },
    {
        "query": "Does the company provide relocation assistance?",
        "expected_source": None,
        "supported": False,
    },
    {
        "query": "Does the company provide transportation benefits?",
        "expected_source": None,
        "supported": False,
    },
    {
        "query": "Is there a relocation allowance?",
        "expected_source": None,
        "supported": False,
    },
]


def get_retrieval_scores(index):
    results = []

    print("\n" + "=" * 80)
    print("PINECONE RETRIEVAL THRESHOLD EVALUATION")
    print("=" * 80)

    for item in TEST_QUERIES:
        query = item["query"]

        retrieval = retrieve(
            index,
            query,
            top_k=5,
        )

        if retrieval.matches:
            top_match = retrieval.matches[0]

            score = float(top_match["score"])

            source = (
                top_match
                .get("metadata", {})
                .get("file_name")
            )

        else:
            score = 0.0
            source = None

        correct_source = (
            item["supported"]
            and source == item["expected_source"]
        )

        results.append(
            {
                "query": query,
                "score": score,
                "source": source,
                "supported": item["supported"],
                "correct_source": correct_source,
            }
        )

        print("\nQuery:", query)
        print("Expected supported:", item["supported"])
        print("Top score:", round(score, 4))
        print("Top source:", source)

    return results


def evaluate_threshold(results, threshold):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for result in results:
        predicted_supported = (
            result["score"] >= threshold
        )

        actual_supported = result["supported"]

        if actual_supported and predicted_supported:
            tp += 1

        elif not actual_supported and not predicted_supported:
            tn += 1

        elif not actual_supported and predicted_supported:
            fp += 1

        elif actual_supported and not predicted_supported:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def main():
    pc = create_pinecone_client()
    index = get_index(pc)

    results = get_retrieval_scores(index)

    thresholds = [
        0.50,
        0.52,
        0.54,
        0.56,
        0.58,
        0.60,
        0.62,
        0.64,
        0.66,
        0.68,
        0.70,
    ]

    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)

    evaluations = []

    for threshold in thresholds:
        evaluation = evaluate_threshold(
            results,
            threshold,
        )

        evaluations.append(evaluation)

        print(
            f"\nThreshold: {threshold:.2f}"
        )
        print(
            f"TP={evaluation['tp']} "
            f"TN={evaluation['tn']} "
            f"FP={evaluation['fp']} "
            f"FN={evaluation['fn']}"
        )
        print(
            f"Accuracy={evaluation['accuracy']:.2%}"
        )
        print(
            f"Precision={evaluation['precision']:.2%}"
        )
        print(
            f"Recall={evaluation['recall']:.2%}"
        )


if __name__ == "__main__":
    main()
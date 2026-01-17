from tests.observability import ProfileBlock, MetricsRegistry
from tests.caching import cacheable
import time
import json


@cacheable(ttl_seconds=60)
def slow_function(x):
    time.sleep(0.1)
    return x * 2


def main():
    with ProfileBlock("Total"):
        with ProfileBlock("Compute"):
            print(slow_function(5))
            print(slow_function(5))  # cache hit

    report = MetricsRegistry.get().generate_report()
    print("\n=== METRICS REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
inspect_weaviate.py

Small Weaviate inspector utility.

Usage examples:
  python tools/inspect_weaviate.py --weaviate http://localhost:8080 --class GameChunk --limit 3 --parse-meta
  python tools/inspect_weaviate.py --weaviate http://localhost:8080 --class GameChunk --where 'source=gamespot:article' --limit 5
  python tools/inspect_weaviate.py --weaviate http://localhost:8080 --class GameChunk --near 'open world,cult' --limit 5
  python tools/inspect_weaviate.py --weaviate http://localhost:8080 --class GameChunk --raw

Notes:
 - The script will always request _additional { id } for each object.
 - If your schema stores `meta` as a JSON string, use --parse-meta to print it parsed.
"""
import argparse
import json
import requests
import sys
import textwrap

DEFAULT_TIMEOUT = 30

def get_schema(weaviate_url):
    url = weaviate_url.rstrip("/") + "/v1/schema"
    r = requests.get(url, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()

def find_class(schema_json, class_name):
    for c in schema_json.get("classes", []):
        if c.get("class") == class_name:
            return c
    return None

def pretty_print_schema_class(c):
    print(f"\nClass: {c.get('class')}\nDescription: {c.get('description')}\n")
    print("Properties:")
    for p in c.get("properties", []):
        name = p.get("name")
        dtype = p.get("dataType")
        desc = p.get("description") or ""
        dtype_str = ", ".join(dtype) if isinstance(dtype, list) else str(dtype)
        flags = []
        if p.get("indexFilterable"): flags.append("filterable")
        if p.get("indexSearchable"): flags.append("searchable")
        if p.get("indexRangeFilters"): flags.append("range")
        print(f"  - {name:18} {dtype_str:20} {' '.join(flags):20} {('- ' + desc) if desc else ''}")
    print("")

def build_safe_graphql(class_name, properties, limit=3, where=None, near=None, order=None):
    select_props = ["_additional { id }"]
    for p in properties:
        name = p.get("name")
        if not name:
            continue
        select_props.append(name)
    select_block = " ".join(select_props)

    where_clause = ""
    if where:
        where_clause = f"where: {{ operator: {where['operator']} path: {json.dumps(where['path'])} valueString: {json.dumps(where['valueString'])} }}"

    near_clause = ""
    if near:
        concepts_json = json.dumps(near['concepts'])
        near_clause = f"nearText: {{ concepts: {concepts_json} }}"
        if near.get("certainty") is not None:
            near_clause = f"nearText: {{ concepts: {concepts_json} certainty: {near['certainty']} }}"

    order_clause = ""
    if order:
        order_clause = f"order: {{ path: {json.dumps(order['path'])}, direction: {order['direction']} }}"

    args = []
    if where_clause:
        args.append(where_clause)
    if near_clause:
        args.append(near_clause)
    if order_clause:
        args.append(order_clause)
    args.append(f"limit: {limit}")
    args_str = ", ".join(args)

    q = f'{{ Get {{ {class_name}({args_str}) {{ {select_block} }} }} }}'
    return q

def run_graphql(weaviate_url, query):
    url = weaviate_url.rstrip("/") + "/v1/graphql"
    payload = {"query": query}
    r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()

def truncate_text(s, max_len=500):
    if s is None:
        return None
    if not isinstance(s, str):
        return s
    if len(s) <= max_len:
        return s
    return s[:max_len] + "... (truncated, len=" + str(len(s)) + ")"

def pretty_print_objects(data_list, properties, parse_meta=False, raw=False):
    if not data_list:
        print("No objects returned.")
        return

    prop_names = [p.get("name") for p in properties if p.get("name")]
    for i, obj in enumerate(data_list, start=1):
        print("\n" + "-"*30)
        print(f"Object #{i}:")
        additional = obj.get("_additional", {})
        if additional and additional.get("id"):
            print("  id:", additional.get("id"))
        for name in prop_names:
            if name not in obj:
                continue
            val = obj[name]
            if name == "meta" and parse_meta and isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    print(f"  {name}:")
                    for k,v in parsed.items():
                        print(f"    {k}: {repr(v)}")
                except Exception:
                    print(f"  {name}: (failed to parse as JSON string) {repr(val[:200])}...")
                continue

            if raw:
                print(f"  {name}: {repr(val)}")
            else:
                if isinstance(val, str) and len(val) > 500:
                    print(f"  {name}: {truncate_text(val, 500)}")
                else:
                    print(f"  {name}: {repr(val)}")
    print("\n" + "="*30 + "\n")

def parse_where_arg(where_arg):
    if not where_arg:
        return None
    if "=" not in where_arg:
        raise ValueError("where must be of form key=value")
    k, v = where_arg.split("=", 1)
    return {"operator": "Equal", "path": [k], "valueString": v}

def main():
    ap = argparse.ArgumentParser(description="Inspect Weaviate class contents and schema.")
    ap.add_argument("--weaviate", required=True, help="Weaviate base URL (e.g. http://localhost:8080)")
    ap.add_argument("--class", dest="classname", default="GameChunk", help="Weaviate class to inspect")
    ap.add_argument("--limit", type=int, default=3, help="How many objects to fetch")
    ap.add_argument("--where", type=str, default=None, help="Simple where clause like 'source=gamespot:article'")
    ap.add_argument("--near", type=str, default=None, help="nearText concepts (comma separated), e.g. 'open world,cult'")
    ap.add_argument("--parse-meta", action="store_true", help="Attempt to parse 'meta' field as JSON string")
    ap.add_argument("--raw", action="store_true", help="Print raw fields without truncation")
    ap.add_argument("--order", type=str, default=None, help="Order clause like 'chunk_index:asc' or 'chunk_index:desc'")
    args = ap.parse_args()

    try:
        schema = get_schema(args.weaviate)
    except Exception as e:
        print("Failed to fetch schema from Weaviate:", e, file=sys.stderr)
        sys.exit(1)

    cls = find_class(schema, args.classname)
    if not cls:
        print(f"Class '{args.classname}' not found. Available classes:")
        for c in schema.get("classes", []):
            print("  -", c.get("class"))
        sys.exit(1)

    pretty_print_schema_class(cls)
    props = cls.get("properties", [])

    where_clause = None
    if args.where:
        try:
            where_clause = parse_where_arg(args.where)
        except Exception as e:
            print("Invalid --where value:", e, file=sys.stderr)
            sys.exit(1)

    near_clause = None
    if args.near:
        concepts = [c.strip() for c in args.near.split(",") if c.strip()]
        if concepts:
            near_clause = {"concepts": concepts}

    order_clause = None
    if args.order:
        if ":" not in args.order:
            print("--order must be like 'chunk_index:asc' or 'chunk_index:desc'", file=sys.stderr)
            sys.exit(1)
        p, d = args.order.split(":", 1)
        d_up = d.strip().lower()
        if d_up not in ("asc", "desc"):
            print("order direction must be 'asc' or 'desc'", file=sys.stderr)
            sys.exit(1)
        order_clause = {"path": [p.strip()], "direction": d_up}

    query = build_safe_graphql(args.classname, props, limit=args.limit, where=where_clause, near=near_clause, order=order_clause)
    print("Constructed GraphQL query:\n")
    print(textwrap.fill(query, width=120))
    print("\nRunning GraphQL query...\n")

    try:
        res = run_graphql(args.weaviate, query)
    except Exception as e:
        print("GraphQL request failed:", e, file=sys.stderr)
        sys.exit(1)

    if args.raw:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    if "errors" in res:
        print("GraphQL returned errors:")
        print(json.dumps(res["errors"], indent=2, ensure_ascii=False))
        sys.exit(1)

    data_list = res.get("data", {}).get("Get", {}).get(args.classname)
    pretty_print_objects(data_list, props, parse_meta=args.parse_meta, raw=args.raw)

if __name__ == "__main__":
    main()
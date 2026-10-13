"""An invented one-cell evaluator integration fixture, never benchmark evidence."""
import argparse
from pathlib import Path

from btsl.evaluate import evaluate
from btsl.io import publish, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-image", default="btsl-evaluator:0.2.0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.output / "invented-prediction"
    (run / "pages").mkdir(parents=True)
    html = "<table><tr><td>Invented fixture</td></tr></table>"
    (run / "pages/fixture.md").write_text(html + "\n", encoding="utf-8")
    publish(run / "READY.json", {"pages": 1, "fixture_only": True,
            "files": {"pages/fixture.md": sha(run / "pages/fixture.md")}})
    gold = args.output / "invented-annotation.json"
    publish(gold, [{"page_info": {"image_path": "fixture.png", "page_no": 1,
                "height": 100, "width": 100, "page_attribute": {}},
        "layout_dets": [{"category_type": "table", "anno_id": 0, "order": 0,
                         "poly": [0, 0, 100, 0, 100, 100, 0, 100], "attribute": {}, "html": html}],
        "extra": {"relation": []}}])
    result = evaluate(run, gold, args.output / "evaluation", args.docker_image,
                      dataset="custom", expected_pages=1, execute=True)
    assert result["arms"]["candidate"]["full_table_teds"] == 100
    assert result["arms"]["candidate"]["structure_teds"] == 100
    print("INVENTED_FIXTURE_INTERFACE_PASS; no project performance claim")


if __name__ == "__main__":
    main()

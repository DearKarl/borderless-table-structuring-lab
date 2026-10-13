"""Create an invented image/Raw pair. This is not benchmark or training data."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "raw").mkdir()
    image = Image.new("RGB", (1100, 750), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=36)
    draw.text((80, 50), "Quarterly inventory", fill="black", font=font)
    for y, row in zip((170, 290, 410, 530), (
        ("Item", "Count", "Cost"), ("Pencils", "12", "24"),
        ("Folders", "8", "40"), ("Notebooks", "10", "50"))):
        for x, value in zip((80, 590, 860), row):
            draw.text((x, y), value, fill="black", font=font)
    image.save(output / "images/inventory.png")
    image.close()
    # A deliberately invented upstream transcription, not a hidden answer file.
    raw = "# Quarterly inventory\n\n<table><tr><td>Item</td><td>Count</td><td>Cost</td></tr><tr><td>Penc1ls</td><td>12</td><td>24</td></tr><tr><td>Folders</td><td>8</td><td>40</td></tr><tr><td>Notebooks</td><td>10</td><td>50</td></tr></table>\n"
    (output / "raw/inventory.md").write_text(raw, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    create(parser.parse_args().output)

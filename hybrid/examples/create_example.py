#!/usr/bin/env python3
"""Create a project-authored toy image and deliberately misspelled raw HTML.

This is an input-format demonstration, not benchmark data or an OCR quality test.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='NEW directory')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    image = Image.new('RGB', (720, 100), 'white')
    draw = ImageDraw.Draw(image)
    for x, text in [(25, 'AnchorA'), (265, '120.5'), (505, 'AnchorB')]:
        draw.text((x, 35), text, fill='black', font_size=28)
    image.save(root / 'table.png')
    with (root / 'raw.html').open('x', encoding='utf-8', newline='') as handle:
        handle.write('<table><tr><td>AnchorA</td><td>12O.5</td><td>AnchorB</td></tr></table>')
    print(str(root.resolve()))


if __name__ == '__main__':
    main()

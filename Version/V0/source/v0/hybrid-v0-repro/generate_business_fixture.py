"""One fixed invented financial table, not a benchmark page or reference answer."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(sys.argv[1]); (root / 'images').mkdir(parents=True); (root / 'raw').mkdir()
    image = Image.new('RGB', (1600, 1200), 'white'); draw = ImageDraw.Draw(image)
    def font(size):
        return ImageFont.load_default(size=size)
    draw.text((120, 110), 'QUARTERLY OPERATING REPORT', font=font(54), fill='black')
    draw.text((120, 210), 'Illustrative figures only. Amounts in thousands.', font=font(32), fill='black')
    xs = [120, 760, 1100, 1480]; ys = [340, 470, 600, 730, 860]
    draw.rectangle((xs[0], ys[0], xs[-1], ys[1]), fill='#eeeeee')
    for x in xs:
        draw.line((x, ys[0], x, ys[-1]), fill='black', width=4)
    for y in ys:
        draw.line((xs[0], y, xs[-1], y), fill='black', width=4)
    rows = [('Metric', 'Q1', 'Q2'), ('Revenue', '120', '150'), ('Costs', '80', '90'), ('Profit', '40', '60')]
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            draw.text((xs[j] + 30, ys[i] + 40), value, font=font(40), fill='black')
    draw.text((120, 980), 'Prepared for a document-processing smoke test.', font=font(30), fill='black')
    page_id = 'synthetic-business-table-001'
    image.save(root / 'images' / (page_id + '.png'))
    raw = '# QUARTERLY OPERATING REPORT\n\nIllustrative figures only. Amounts in thousands.\n\n'
    raw += '<table>' + ''.join('<tr>' + ''.join('<td>' + value + '</td>' for value in row) + '</tr>' for row in rows) + '</table>\n\n'
    raw += 'Prepared for a document-processing smoke test.\n'
    (root / 'raw' / (page_id + '.md')).write_text(raw, encoding='utf-8')


if __name__ == '__main__':
    main()

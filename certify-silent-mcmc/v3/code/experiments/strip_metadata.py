"""Strip metadata from PNG and PDF files (in place), recursively under the given paths.
PNG: re-encode pixels only (drops all ancillary text, EXIF and provenance chunks).
PDF: remove the document-info dictionary and XMP metadata stream.
Usage: python strip_metadata.py <dir-or-file> [...]"""
import os, sys
from PIL import Image
import pikepdf


def strip_png(p):
    im = Image.open(p)
    im.load()
    clean = Image.new(im.mode, im.size)
    clean.putdata(list(im.getdata())) if im.mode not in ('RGB', 'RGBA', 'L', 'LA', 'P') else None
    if im.mode in ('RGB', 'RGBA', 'L', 'LA', 'P'):
        clean = im.copy()
        clean.info = {}
    if im.mode == 'P' and 'transparency' in im.info:
        clean = im.convert('RGBA')
    clean.save(p, format='PNG', pnginfo=None)


def strip_pdf(p):
    with pikepdf.open(p, allow_overwriting_input=True) as pdf:
        if '/Metadata' in pdf.Root:
            del pdf.Root['/Metadata']
        pdf.trailer['/Info'] = pikepdf.Dictionary()
        pdf.save(p)


def walk(path):
    if os.path.isfile(path):
        yield path
    else:
        for r, _, fs in os.walk(path):
            for f in fs:
                yield os.path.join(r, f)


n = 0
for arg in sys.argv[1:]:
    for f in walk(arg):
        ext = f.lower().rsplit('.', 1)[-1]
        if ext == 'png':
            strip_png(f); n += 1
        elif ext == 'pdf':
            strip_pdf(f); n += 1
print('stripped', n, 'files')

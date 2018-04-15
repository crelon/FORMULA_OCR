# -*- coding: utf-8 -*-

import io
import time

from wand.image import Image
from wand.color import Color
from PyPDF2 import PdfFileReader, PdfFileWriter

memo = {}


def getPdfReader(filename):
    reader = memo.get(filename, None)
    if reader is None:
        reader = PdfFileReader(filename, strict=False)
        memo[filename] = reader
    return reader


def _run_convert(filename, page, res=120):
    idx = page + 1
    temp_time = time.time() * 1000
    # 由于每次转换的时候都需要重新将整个PDF载入内存，所以这里使用内存缓存
    pdfile = getPdfReader(filename)
    pageObj = pdfile.getPage(page)
    dst_pdf = PdfFileWriter()
    dst_pdf.addPage(pageObj)

    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)

    img = Image(file=pdf_bytes, resolution=res)
    img.format = 'png'
    img.compression_quality = 90
    img.background_color = Color("white")
    # 保存图片
    img_path = '%s%d.png' % (filename[:filename.rindex('.')], idx)
    img.save(filename=img_path)
    img.destroy()
    img = None
    pdf_bytes = None
    dst_pdf = None
    print('convert page %d cost time %d' % (idx,
                                            (time.time() * 1000 - temp_time)))


# if __name__ == '__main__':
#     _run_convert('demo.pdf', 0)

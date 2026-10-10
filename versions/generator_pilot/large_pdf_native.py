"""Open cached PDF bytes with the same backend as pinned TeleOCR composition."""


def open_native_pdf(pdf_bytes):
    import TeleOCR.config as config
    if config.PDF_TOOLS == 'pypdfium2':
        import pypdfium2
        return pypdfium2.PdfDocument(pdf_bytes)
    if config.PDF_TOOLS == 'PyMuPDF':
        import fitz
        return fitz.open(stream=pdf_bytes, filetype='pdf')
    raise ValueError('Unsupported pinned TeleOCR PDF backend: ' + str(config.PDF_TOOLS))

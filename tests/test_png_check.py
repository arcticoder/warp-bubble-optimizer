import os
from tools.png_check import is_png

def test_is_png_detects_header(tmp_path):
    p = tmp_path / 'x.png'
    p.write_bytes(b'\x89PNG\r\n\x1a\n' + b'junk')
    assert is_png(str(p)) is True

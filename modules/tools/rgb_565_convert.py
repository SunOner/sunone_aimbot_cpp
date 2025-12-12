from pathlib import Path
import struct, sys, textwrap
from PIL import Image

SIZES = [(128,  80), (128, 160)] # (W, H)

def rgb888_to_565(r, g, b):
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) |  (b >> 3)

def image_to_rgb565(img, w, h):
    img = img.convert('RGB').resize((w, h), Image.LANCZOS)
    out = bytearray()
    for y in range(h):
        for x in range(w):
            out += struct.pack('<H', rgb888_to_565(*img.getpixel((x, y))))
    return out

def write_h_word(arr_name, data16, path):
    with open(path, 'w', encoding='utf-8') as h:
        h.write(f'unsigned char {arr_name}[{len(data16)//2}] = {{\n  ')
        for i, (hi, lo) in enumerate(zip(data16[1::2], data16[::2])):  # little-endian -> word
            if i and i % 12 == 0:
                h.write('\n  ')
            h.write(f'0x{lo:02X}{hi:02X}, ')
        h.write('\n};\n')

def write_h_byte(arr_name, data8, path, put_hdr_comment=False):
    with open(path, 'w', encoding='utf-8') as h:
        hdr = (' /* 0X00,0X10,0X80,0X00,0XA0,0X00,0X01,0X1B, */' 
               if put_hdr_comment else '')
        h.write(f'unsigned char {arr_name}[{len(data8)}] = {{{hdr}\n  ')
        for i, b in enumerate(data8):
            if i and i % 16 == 0:
                h.write('\n  ')
            h.write(f'0X{b:02X}, ')
        h.write('\n};\n')

def main(src_img):
    img = Image.open(src_img)
    base = Path(src_img).stem

    for w, h in SIZES:
        rgb565 = image_to_rgb565(img, w, h)
        raw_path = f'{base}_{w}x{h}.raw'
        Path(raw_path).write_bytes(rgb565)
        assert len(rgb565) == w*h*2, f'Bad size for {w}x{h}'

        if h == 80:
            write_h_word(f'gImage_{w}x{h}', rgb565, f'gImage_{w}x{h}.h')
        else:
            write_h_byte(f'gImage_{w}x{h}', rgb565, f'gImage_{w}x{h}.h',
                         put_hdr_comment=True)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python rgb565_convert.py <input_img>')
        sys.exit(1)
    main(sys.argv[1])
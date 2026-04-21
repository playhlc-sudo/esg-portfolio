"""
PWA 아이콘 생성 스크립트
실행: python generate_icons.py
"""
import os

# 아이콘을 생성할 사이즈들
SIZES = [72, 96, 128, 144, 152, 192, 512]

def create_svg_icon():
    """SVG 아이콘 생성"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#FF4B4B"/>
      <stop offset="100%" style="stop-color:#FF6B6B"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="80" fill="url(#bg)"/>
  <text x="256" y="200" font-family="Arial, sans-serif" font-size="120" font-weight="bold" fill="white" text-anchor="middle">ESG</text>
  <text x="256" y="340" font-family="Arial, sans-serif" font-size="60" fill="white" text-anchor="middle" opacity="0.9">Portfolio</text>
  <path d="M140 380 L200 420 L260 360 L320 400 L380 340" stroke="white" stroke-width="12" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''
    return svg

def main():
    icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
    os.makedirs(icons_dir, exist_ok=True)

    # SVG 아이콘 저장
    svg_content = create_svg_icon()
    svg_path = os.path.join(icons_dir, 'icon.svg')
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"SVG 아이콘 생성: {svg_path}")

    # PNG 변환 시도 (Pillow + cairosvg 필요)
    try:
        import cairosvg
        for size in SIZES:
            png_path = os.path.join(icons_dir, f'icon-{size}.png')
            cairosvg.svg2png(bytestring=svg_content.encode(), write_to=png_path,
                          output_width=size, output_height=size)
            print(f"PNG 아이콘 생성: icon-{size}.png")
        print("\n모든 아이콘이 생성되었습니다!")
    except ImportError:
        print("\n[알림] PNG 변환을 위해 cairosvg가 필요합니다:")
        print("  pip install cairosvg")
        print("\n또는 온라인 변환기를 사용하세요:")
        print("  https://cloudconvert.com/svg-to-png")
        print(f"\n  SVG 파일 위치: {svg_path}")

        # Pillow만으로 간단한 아이콘 생성 시도
        try:
            from PIL import Image, ImageDraw, ImageFont
            print("\nPillow로 대체 아이콘 생성 중...")

            for size in SIZES:
                img = Image.new('RGB', (size, size), '#FF4B4B')
                draw = ImageDraw.Draw(img)

                # 간단한 텍스트 추가
                text = "ESG"
                font_size = size // 4
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (size - text_width) // 2
                y = (size - text_height) // 2
                draw.text((x, y), text, fill='white', font=font)

                png_path = os.path.join(icons_dir, f'icon-{size}.png')
                img.save(png_path)
                print(f"  생성됨: icon-{size}.png")

            print("\n기본 아이콘이 생성되었습니다!")
        except ImportError:
            print("\nPillow 설치 후 다시 시도하세요:")
            print("  pip install Pillow")

if __name__ == '__main__':
    main()

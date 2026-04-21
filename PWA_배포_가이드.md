# ESG 포트폴리오 PWA 배포 가이드

## 배포 순서

### 1단계: 아이콘 생성
```bash
cd pwa
pip install Pillow
python generate_icons.py
```

### 2단계: GitHub 저장소 생성
1. [github.com](https://github.com) 에서 새 저장소 생성
2. 저장소 이름: `esg-portfolio` (또는 원하는 이름)
3. Public으로 설정

### 3단계: 코드 업로드
```bash
cd C:\Users\gc411\Desktop\hyun
git init
git add .
git commit -m "Initial commit: ESG Portfolio Backtest"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/esg-portfolio.git
git push -u origin main
```

### 4단계: Streamlit Cloud 배포
1. [share.streamlit.io](https://share.streamlit.io) 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 저장소 선택: `YOUR_USERNAME/esg-portfolio`
5. Main file path: `app.py`
6. "Deploy!" 클릭
7. 배포 완료 후 URL 복사 (예: `https://your-app.streamlit.app`)

### 5단계: PWA 설정 업데이트
`pwa/index.html` 파일에서 아래 부분 수정:
```javascript
const STREAMLIT_URL = 'https://your-app.streamlit.app';  // 실제 URL로 변경
```

### 6단계: PWA 호스팅 (GitHub Pages)
1. GitHub 저장소 Settings → Pages
2. Source: Deploy from a branch
3. Branch: `main` / `pwa` 폴더 선택
4. Save

또는 별도 저장소로 PWA만 배포:
```bash
cd pwa
git init
git add .
git commit -m "PWA files"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/esg-pwa.git
git push -u origin main
```
GitHub Pages 설정 후 `https://YOUR_USERNAME.github.io/esg-pwa` 에서 접속

---

## 핸드폰에 앱 설치하기

### Android
1. Chrome으로 PWA 주소 접속
2. 메뉴(⋮) → "홈 화면에 추가" 또는
3. 자동으로 뜨는 "앱 설치" 배너 클릭

### iPhone/iPad
1. Safari로 PWA 주소 접속
2. 공유 버튼(□↑) 클릭
3. "홈 화면에 추가" 선택
4. "추가" 클릭

---

## 폴더 구조
```
hyun/
├── app.py                 # 메인 Streamlit 앱
├── backtest.py            # 백테스트 엔진
├── rebalance_engine.py    # 리밸런싱 엔진
├── news_engine.py         # 뉴스 분석
├── ml_engine.py           # ML 예측
├── regime_engine.py       # 레짐 분석
├── risk_engine.py         # 리스크 관리
├── cross_asset_engine.py  # 크로스 에셋
├── requirements.txt       # 패키지 목록
├── user_settings.json     # 사용자 설정
└── pwa/
    ├── index.html         # PWA 메인 페이지
    ├── manifest.json      # PWA 설정
    ├── service-worker.js  # 서비스 워커
    ├── generate_icons.py  # 아이콘 생성
    └── icons/             # 앱 아이콘들
```

---

## 문제 해결

### "앱 설치" 옵션이 안 보여요
- HTTPS가 필수입니다 (GitHub Pages는 자동 HTTPS)
- manifest.json이 올바르게 연결되어 있는지 확인
- Chrome DevTools → Application → Manifest 에서 확인

### Streamlit 앱이 로딩 안 돼요
- Streamlit Cloud 배포 상태 확인
- STREAMLIT_URL이 정확한지 확인
- 브라우저 콘솔에서 에러 확인 (F12)

### 아이콘이 안 보여요
- icons 폴더에 PNG 파일들이 있는지 확인
- generate_icons.py 실행 확인

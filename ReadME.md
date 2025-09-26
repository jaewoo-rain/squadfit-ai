# (프로젝트 루트: docker-compose.yml이 있는 곳)
# 1) 이미지 빌드 (aiortc만 빌드, coturn은 pull)
docker compose build

# 2) 컨테이너 띄우기 (필요하면 이미지 다시 빌드)
docker compose up -d    # 또는 docker compose up --build -d

# 3) 상태 확인
docker compose ps
docker compose logs -f aiortc
docker compose logs -f coturn
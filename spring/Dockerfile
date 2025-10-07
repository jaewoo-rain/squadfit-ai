# Dockerfile (Spring Boot JAR 실행용)
FROM eclipse-temurin:17-jre-alpine

# 보안상 non-root 실행
RUN addgroup -S app && adduser -S app -G app
USER app

# 작업 디렉터리
WORKDIR /app

# ⬇️ 실제 JAR 파일명에 맞추세요 (현재: squadfit-0.0.1-SNAPSHOT.jar)
COPY squadfit-0.0.1-SNAPSHOT.jar app.jar

# 런타임 옵션 & 타임존
ENV JAVA_OPTS="-XX:MaxRAMPercentage=75 -Duser.timezone=Asia/Seoul"
ENV SERVER_PORT=8080

# 컨테이너 내부 포트
EXPOSE 8080

# 헬스체크: /turn/credentials가 동작해야 healthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD wget -qO- http://localhost:${SERVER_PORT}/turn/credentials || exit 1

# 실행
ENTRYPOINT ["/bin/sh","-c","java $JAVA_OPTS -jar app.jar --server.port=${SERVER_PORT}"]

# docker-compose -f docker-compose.spring.yml up -d --build

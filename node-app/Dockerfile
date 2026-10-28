FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=production \
    NODE_PORT=3000 \
    PYTHON_RAG_URL=http://python-rag:8000

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY server.js ./

EXPOSE 3000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=5 \
    CMD wget -qO- http://127.0.0.1:3000/health || exit 1

CMD ["node", "server.js"]
